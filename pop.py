import torch

class POPLoss(torch.nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def forward(self, x, y):
        loss = torch.nn.functional.cross_entropy(x, y)
        
        with torch.no_grad():
            rejected = torch.multinomial(torch.softmax(x / 0.7, dim=-1), 1).squeeze(-1)
            chosen = y

        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        chosen_log_probs = torch.gather(log_probs, -1, chosen.unsqueeze(-1)).squeeze(-1)
        rejected_log_probs = torch.gather(log_probs, -1, rejected.unsqueeze(-1)).squeeze(-1)
        
        log_odds = (chosen_log_probs - rejected_log_probs) - (torch.log1p(-torch.exp(chosen_log_probs)) - torch.log1p(-torch.exp(rejected_log_probs))) 

        sigratio = torch.sigmoid(log_odds)
        loss = self.beta * torch.log(sigratio) + loss
        return loss.mean()
