# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # dist = dist.to(y.dtype)
    dist.addmm_( x, y.t(),alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    # dist = dist # for numerical stability
    # print(dist)
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    # print(dist_an)
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds


    return dist_ap, dist_an
################################
def New_dist(embeddings):
	num=len(embeddings)
	d=[]
	d_min=[]
	for i in range (num):
		d.append([])
		for j in range(num):
			d[i].append(euclidean_dist(embeddings[i],embeddings[j]))
	for i in range (num):
		d_min.append([])
		d_min[i] = torch.where(d[0][i] < d[i][1], d[0][i], d[i][1])
		# print(d_min[i].shape)
		if num >2:
			j=2
			while j<num:
				d_min[i] = torch.where(d_min[i] < d[i][j], d_min[i], d[i][j])
				j+=1

	distances = 0
	for i in range(num):
		distances+=d_min[i]
	# distances = d_min.sum()
	return distances

def batch_all(dist_mat,labels):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    # print(is_neg.shape)
    # print(dist_mat[is_pos].shape)
    # print(dist_mat*is_neg)
    return torch.triu(dist_mat*is_pos,1),torch.triu(dist_mat*is_neg,1)
######################################
class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None,new_dist = False,if_view = False):
        self.margin = margin
        self.new_dist = new_dist
        self.if_view = if_view
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, view_mat = None):
        if not self.new_dist:
            dist_mat = euclidean_dist(global_feat, global_feat)
        else:
            dist_mat = New_dist(global_feat)
        if self.if_view :
            dist_sv = dist_mat*view_mat
            dist_dv = dist_mat - dist_sv
            dist_ap_sv,dist_an_sv = batch_all(dist_sv,labels)
            dist_ap_dv, dist_an_dv = batch_all(dist_dv, labels)
            y1 = dist_mat.new().resize_as_(dist_mat).fill_(1)
        # dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        dist_ap, dist_an = batch_all(dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            if self.if_view :
                loss_sv = self.ranking_loss(dist_an_sv,dist_ap_sv,y1)
                loss_dv = self.ranking_loss(dist_an_dv,dist_ap_dv,y1)
                loss_cv = self.ranking_loss(dist_an_sv,dist_ap_dv,y1)
                loss_tri = self.ranking_loss(dist_an, dist_ap, y)
                loss = loss_cv + loss_dv + loss_sv + loss_tri
            else:
                loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


if __name__ == "__main__":
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputs=torch.randn(8,156).to(device)
    labels=torch.randn(8).to(device)
    # for i in range(2):
    #     inputs.append(torch.randn(5,256).to(device))
    tripletloss = TripletLoss(margin=0.3)
    output = tripletloss(inputs,labels)[0]
    print(output)