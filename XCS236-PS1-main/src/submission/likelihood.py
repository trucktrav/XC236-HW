import torch
import torch.nn as nn
from click.core import F


def log_likelihood(model, text):
    """
    Compute the log-likelihoods for a string `text`
    :param model: The GPT-2 model
    :param texts: A tensor of shape (1, T), where T is the length of the text
    :return: The log-likelihood. It should be a Python scalar.
        NOTE: for simplicity, you can ignore the likelihood of the first token in `text`.
    """

    with torch.no_grad():
        ## TODO:
        ##  1) Compute the logits from `model`;
        ##  2) Return the log-likelihood of the `text` string. It should be a Python scalar.
        ##      NOTE: for simplicity, you can ignore the likelihood of the first token in `text`
        ##      Hint: Checkout Pytorch softmax: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
        ##                     Pytorch negative log-likelihood: https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
        ##                     Pytorch Cross-Entropy Loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        ## 
        ## The problem asks us for (positive) log likelihood, which would be equivalent to the negative of a negative log likelihood. 
        ## Cross Entropy Loss is equivalent applying LogSoftmax on an input, followed by NLLLoss. Use reduction 'sum'.
        ## 
        ## Hint: Implementation should only takes 3~7 lines of code.
        
        ### START CODE HERE ###
        logits, newpast = model(text, None)
        # probs = torch.softmax(logits[:, -1, :], dim=-1)
        celoss = nn.CrossEntropyLoss(reduction='sum')

        i_logits = logits[0, :-1, :]
        i_labels = text[0, 1:]
        # loss = -1 * celoss(i_logits.view(-1, i_logits.size(-1)), i_labels.view(-1))
        # loss = -1* celoss(logits[:, -1, :], torch.tensor([0]))

        # loss_fn = nn.NLLLoss()
        # loss = loss_fn(text, probs)
        return -1 * celoss(i_logits, i_labels)
        ### END CODE HERE ###
        raise NotImplementedError
