import torch

def moe_topk_routing(
    router_logits: torch.Tensor,
    expert_outputs: torch.Tensor,
    k: int
) -> torch.Tensor:

    router, indices = torch.topk(router_logits, k=k, dim=-1)

    weights = torch.softmax(router, dim=-1)

    selected = torch.gather(
        expert_outputs,
        dim=1,
        index=indices.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[-1])
    )

    output = (selected * weights.unsqueeze(-1)).sum(dim=1)

    return output