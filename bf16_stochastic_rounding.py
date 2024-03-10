import math
from typing import List, Optional, Union

import torch
from torch import Tensor
from torch.optim.optimizer import _use_grad_for_differentiable


def copy_stochastic_(target: Tensor, source: Tensor):
    """
    copies source into target using stochastic rounding

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
    """
    # create a random 16 bit integer
    result = torch.randint_like(
        source,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
    )

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))

    del result


def add_stochastic_(input: Tensor, other: Tensor, alpha: float = 1.0):
    """
    adds other to input using stochastic rounding

    Args:
        input: the input tensor with dtype=bfloat16
        other: the other tensor
        alpha: a multiplier for other
    """
    if other.dtype == torch.float32:
        result = other.clone()
    else:
        result = other.to(dtype=torch.float32)

    result.add_(input, alpha=alpha)
    copy_stochastic_(input, result)


def addcdiv_stochastic_(input: Tensor, tensor1: Tensor, tensor2: Tensor, value: float = 1.0):
    """
    adds (tensor1 / tensor2 * value) to input using stochastic rounding

    Args:
        input: the input tensor with dtype=bfloat16
        tensor1: the numerator tensor
        tensor2: the denominator tensor
        value: a multiplier for tensor1/tensor2
    """
    if input.dtype == torch.float32:
        result = input.clone()
    else:
        result = input.to(dtype=torch.float32)

    result.addcdiv_(tensor1, tensor2, value=value)
    copy_stochastic_(input, result)



#################
## ADAMW
#################

def _single_tensor_adamw(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: Union[Tensor, float],
        weight_decay: float,
        eps: float,
        maximize: bool,
        capturable: bool,
        differentiable: bool,
):

    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch._utils.is_compiling() and capturable:
            assert (
                (param.is_cuda and step_t.is_cuda) or (param.is_xla and step_t.is_xla)
            ), "If capturable=True, params and state_steps must be CUDA or XLA tensors."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable or differentiable:
            step = step_t

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (
                    max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)
            else:
                denom = (
                    exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            step = step_t.item()

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1

            if isinstance(bias_correction2, torch.Tensor):
                bias_correction2_sqrt = bias_correction2.sqrt()
            else:
                bias_correction2_sqrt = math.sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            if param.dtype == torch.bfloat16:
                addcdiv_stochastic_(param, exp_avg, denom, value=-step_size)
            else:
                param.addcdiv_(exp_avg, denom, value=-step_size)

        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])


@_use_grad_for_differentiable
def step_adamw(self, closure=None):
    """Performs a single optimization step.

    Args:
        closure (Callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    self._cuda_graph_capture_health_check()

    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps = []
        amsgrad = group["amsgrad"]
        beta1, beta2 = group["betas"]

        self._init_group(
            group,
            params_with_grad,
            grads,
            amsgrad,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
        )

        _single_tensor_adamw(
            params=params_with_grad,
            grads=grads,
            exp_avgs=exp_avgs,
            exp_avg_sqs=exp_avg_sqs,
            max_exp_avg_sqs=max_exp_avg_sqs,
            state_steps=state_steps,
            grad_scale=getattr(self, "grad_scale", None),
            found_inf=getattr(self, "found_inf", None),
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group["maximize"],
            capturable=group["capturable"],
            differentiable=group["differentiable"]
        )

    return loss


#################
## ADAFACTOR
#################

@torch.no_grad()
def step_adafactor(self, closure=None):
    """
    Performs a single optimization step

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.dtype in {torch.float16, torch.bfloat16}:
                grad = grad.float()
            if grad.is_sparse:
                raise RuntimeError("Adafactor does not support sparse gradients.")

            state = self.state[p]
            grad_shape = grad.shape

            factored, use_first_moment = self._get_options(group, grad_shape)
            # State Initialization
            if len(state) == 0:
                state["step"] = 0

                if use_first_moment:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                if factored:
                    state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                    state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                else:
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                state["RMS"] = 0
            else:
                if use_first_moment:
                    state["exp_avg"] = state["exp_avg"].to(grad)
                if factored:
                    state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                    state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                else:
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

            p_data_fp32 = p
            if p.dtype in {torch.float16, torch.bfloat16}:
                p_data_fp32 = p_data_fp32.float()

            state["step"] += 1
            state["RMS"] = self._rms(p_data_fp32)
            lr = self._get_lr(group, state)

            beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
            update = (grad ** 2) + group["eps"][0]
            if factored:
                exp_avg_sq_row = state["exp_avg_sq_row"]
                exp_avg_sq_col = state["exp_avg_sq_col"]

                exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
                exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))

                # Approximation of exponential moving average of square of gradient
                update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                update.mul_(grad)
            else:
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                update = exp_avg_sq.rsqrt().mul_(grad)

            update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
            update.mul_(lr)

            if use_first_moment:
                exp_avg = state["exp_avg"]
                exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                update = exp_avg

            if group["weight_decay"] != 0:
                p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

            p_data_fp32.add_(-update)

            if p.dtype == torch.bfloat16:
                copy_stochastic_(p, p_data_fp32)
            elif p.dtype == torch.float16:
                p.copy_(p_data_fp32)

    return loss
