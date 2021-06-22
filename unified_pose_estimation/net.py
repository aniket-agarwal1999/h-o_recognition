import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from cfg import parameters

class UnifiedNetwork(nn.Module):

    def __init__(self):

        super(UnifiedNetwork, self).__init__()

        self.num_hand_control_points = parameters.num_hand_control_points
        self.num_object_control_points = parameters.num_object_control_points
        self.num_actions = parameters.num_actions
        self.num_objects = parameters.num_objects
        self.depth_discretization = parameters.depth_discretization
        
        model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-2])

        self.hand_vector_size = 3 * self.num_hand_control_points + 1 + self.num_actions      ### This is present in every cell, basically all preds related to hand pose and action
        self.object_vector_size = 3 * self.num_object_control_points + 1 + self.num_objects     ### This is present in every cell, basically all preds related to object pose and object class
        self.target_channel_size = self.depth_discretization * ( self.hand_vector_size + self.object_vector_size )

        # prediction layers
        self.conv = nn.Conv2d(512, self.target_channel_size, (3,3), padding=1, bias=True)

        # losses
        self.setup_losses()
    
    def setup_losses(self):

        self.action_ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.object_ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x):

        # split it into different types of data
        height, width = x.size()[2:]
        
        assert height == width
        assert height % 32 == 0
        
        target_height, target_width = height // 32, width // 32

        x = self.features(x)
        x = self.conv(x).view(-1, self.hand_vector_size + self.object_vector_size, self.depth_discretization, target_height, target_width)
        
        pred_v_h = x[:, :self.hand_vector_size, :, :, :]
        pred_v_o = x[:, self.hand_vector_size:, :, :, :]

        # hand specific predictions
        pred_hand_pose = pred_v_h[:, :3*self.num_hand_control_points, :, :, :]
        pred_hand_pose = pred_hand_pose.view(-1, 21, 3, 5, 13, 13)
        pred_hand_pose_root = torch.sigmoid(pred_hand_pose[:, 0, :, :, :, :].unsqueeze(1))      ### Since the root join is to be constrained from 0 to 1
        pred_hand_pose_without_root = pred_hand_pose[:, 1:, :, :, :, :]      ### There does not need to be a constraint here
        pred_hand_pose = torch.cat((pred_hand_pose_root, pred_hand_pose_without_root), 1).view(-1, 63, 5, 13, 13)
        pred_action_prob = pred_v_h[:, 3*self.num_hand_control_points:-1, :, :, :]        ### The action prob tensor
        pred_hand_conf = torch.sigmoid(pred_v_h[:, -1, :, :, :])        ### The confidence value

        # object specific predictions
        ### Same as above but just for object pose and object class
        pred_object_pose = pred_v_o[:, :3*self.num_object_control_points, :, :, :]
        pred_object_pose = pred_object_pose.view(-1, 21, 3, 5, 13, 13)
        pred_object_pose_root = torch.sigmoid(pred_object_pose[:, 0, :, :, :, :].unsqueeze(1))
        pred_object_pose_without_root = pred_object_pose[:, 1:, :, :, :, :]
        pred_object_pose = torch.cat((pred_object_pose_root, pred_object_pose_without_root), 1).view(-1, 63, 5, 13, 13)
        pred_object_prob = pred_v_o[:, 3*self.num_object_control_points:-1, :, :, :]
        pred_object_conf = torch.sigmoid(pred_v_o[:, -1, :, :, :])

        return pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf
        
    def total_loss(self, pred, true):

        pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf = pred
        true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask = true
        
        total_pose_loss = self.pose_loss(pred_hand_pose, true_hand_pose, hand_mask) + self.pose_loss(pred_object_pose, true_object_pose, object_mask)
        total_conf_loss = self.conf_loss(pred_hand_conf, pred_hand_pose, true_hand_pose, hand_mask) + self.conf_loss(pred_object_conf, pred_object_pose, true_object_pose, object_mask)
        total_action_loss = self.action_loss(pred_action_prob, true_action_prob, hand_mask)
        total_object_loss = self.object_loss(pred_object_prob, true_object_prob, object_mask)

        total_loss = total_pose_loss + total_action_loss + total_object_loss + total_conf_loss

        return total_loss

    def pose_loss(self, pred, true, mask):

        pred = pred.view(-1, 21, 3, 5, 13, 13)
        true = true.view(-1, 21, 3, 5, 13, 13)
        masked_pose_loss = torch.mean(torch.sum(mask * torch.sum(torch.mul(pred - true, pred - true), dim=[1,2]), dim=[1,2,3]))
        return masked_pose_loss
    
    def conf_loss(self, pred_conf, pred, true, mask):
        
        pred = pred.view(-1, 21, 3, 5, 13, 13)
        true = true.view(-1, 21, 3, 5, 13, 13)

        ### Since we are calculating the whole distance in 'image spcae', so just untransforming to a different space using same transforms as in dataset.py
        pred_pixel_x = pred[:, :, 0, :, :, :].unsqueeze(2) * 32 * (1920. / 416)
        pred_pixel_y = pred[:, :, 1, :, :, :].unsqueeze(2) * 32 * (1080. / 416)
        pred_depth = pred[:, :, 2, :, :, :] * 15 * 10

        ### Since we are calculating the whole distance in 'image spcae', so just untransforming to a different space using same transforms as in dataset.py
        true_pixel_x = true[:, :, 0, :, :, :].unsqueeze(2) * 32 * (1920. / 416)
        true_pixel_y = true[:, :, 1, :, :, :].unsqueeze(2) * 32 * (1080. / 416)
        true_depth = true[:, :, 2, :, :, :] * 15 * 10

        pixel_x_distance = torch.mul(pred_pixel_x - true_pixel_x, pred_pixel_x - true_pixel_x)
        pixel_y_distance = torch.mul(pred_pixel_y - true_pixel_y, pred_pixel_y - true_pixel_y)
        pixel_distance = torch.sqrt(pixel_x_distance + pixel_y_distance).squeeze(2)
        depth_distance = torch.sqrt(torch.mul(pred_depth - true_depth, pred_depth - true_depth))
        
        # threshold
        pixel_distance_mask = (pixel_distance < parameters.pixel_threshold).type(torch.cuda.FloatTensor)
        depth_distance_mask = (depth_distance < parameters.depth_threshold).type(torch.cuda.FloatTensor)

        pixel_distance = pixel_distance / (32 * 416 / 1920.)       ### going again from image space to cell space
        depth_disrance = depth_distance / (15 * 10.)

        pixel_distance = torch.from_numpy(pixel_distance.cpu().data.numpy()).cuda()
        depth_distance = torch.from_numpy(depth_distance.cpu().data.numpy()).cuda()

        pixel_conf = torch.exp(parameters.sharpness * (1 - pixel_distance / parameters.pixel_threshold)) / torch.exp(parameters.sharpness * (1 - torch.zeros(pixel_distance.size()).cuda()))
        depth_conf = torch.exp(parameters.sharpness * (1 - depth_distance / parameters.depth_threshold)) / torch.exp(parameters.sharpness * (1 - torch.zeros(depth_distance.size()).cuda()))

        pixel_conf = torch.mean(pixel_distance_mask * pixel_conf, dim=1)
        depth_conf = torch.mean(depth_distance_mask * depth_conf, dim=1)

        ### Till here it was just calculation of confidence value using the pose features and nothing else, ahead we calculate the error between true and pred

        true_conf = 0.5 * (pixel_conf + depth_conf)
        squared_conf_error = torch.mul(pred_conf - true_conf, pred_conf - true_conf)
        exist_conf_error = torch.mean(torch.sum(mask * squared_conf_error, dim=[1,2,3]))

        true_conf = torch.zeros(pred_conf.size()).cuda()
        squared_conf_error = torch.mul(pred_conf - true_conf, pred_conf - true_conf)
        no_exist_conf_error = torch.mean(torch.sum((1 - mask) * squared_conf_error, dim=[1,2,3]))

        return 5 * exist_conf_error + 0.1 * no_exist_conf_error
        
    def action_loss(self, pred, true, mask):
        action_ce_loss = self.action_ce_loss(pred, true)
        return torch.mean(torch.sum(mask * action_ce_loss, dim=[1,2,3]))

    def object_loss(self, pred, true, mask):
        object_ce_loss = self.object_ce_loss(pred, true)
        return torch.mean(torch.sum(mask * object_ce_loss, dim=[1,2,3]))


class InteractionRNN(nn.Module):

    def __init__(self):
        super(InteractionRNN, self).__init__()

        self.hand_pose_dim = parameters.num_hand_control_points * 3
        self.object_pose_dim = parameters.num_object_control_points * 3
        self.fc_hidden = parameters.fc_hidden
        self.lstm_layers = parameters.lstm_layers
        self.lstm_hidden = parameters.lstm_hidden
        self.num_actions = parameters.num_actions
        self.num_objects = parameters.num_objects

        self.fc = nn.Linear(self.hand_pose_dim + self.object_pose_dim, self.fc_hidden)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(self.fc_hidden, self.lstm_hidden, self.lstm_layers, batch_first=True)   ###(batch, seq_len, input_size)

        self.fc_final = nn.Linear(self.lstm_hidden, self.num_actions + self.num_objects)

        self.setup_losses()

    def init_hidden_state(self, batch_size):
        h1 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to('cuda').float()
        h2 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to('cuda').float()
        return (h1, h2)

        # h1 = torch.empty(self.n_layers * 2, batch_size, self.hidden_dim).float()
        # h1 = nn.init.orthogonal_(h1)
        # h1 = h1.requires_grad_().to('cuda')
        # h2 = torch.empty(self.n_layers * 2, batch_size, self.hidden_dim).float()
        # h2 = nn.init.orthogonal_(h2)
        # h2 = h2.requires_grad_().to('cuda')
        # return (h1, h2)

    def setup_losses(self):
        self.action_ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.object_ce_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        batch_size = 1     ### The only hardcoded part since here we have to keep batch size as 1

        h = self.init_hidden_state(batch_size)
        out = self.fc(x)
        out = self.relu(out)
        out = out.unsqueeze(0)

        out, _ = self.lstm(out, h)
        out = out.squeeze()           ### Assuming the batch size is 0 since this will result in (seq_len, hidden_dim)
        # print('LSTM out shape: ', out.shape)

        out = self.fc_final(out)

        return out[:, :self.num_actions], out[:, self.num_objects:]

    def total_loss(self, pred, gt_action, gt_object):
        pred_action, pred_object = pred

        # print('Shape pred_action: ', pred_action.shape)
        # print('Shape pred_object: ', pred_object.shape)
        # print('gt_action shape: ', gt_action.shape)
        action_loss = self.action_ce_loss(pred_action, gt_action)
        object_loss = self.object_ce_loss(pred_object, gt_object)

        total_loss = action_loss + object_loss
        # print('total_loss shape: ', total_loss)
        return total_loss



if __name__ == '__main__':

    model = UnifiedNetwork()
    model = model.cuda()
    x = torch.randn(32, 3, 416, 416).cuda()

    ### idk why this is here, no use whatsoever
    # true = torch.randn(32, 76, 5, 13, 13), torch.randn(32, 74, 5, 13, 13)

    pred =  model(x)
    
    true_hand_pose = torch.randn(32, 3 * parameters.num_hand_control_points, 5, 13, 13).cuda()
    true_action_prob = torch.empty(32, 5, 13, 13, dtype=torch.long).random_(parameters.num_actions).cuda()
    hand_mask = torch.zeros(5, 13, 13, dtype=torch.float32).cuda()
    hand_mask[0, 0, 0] = 1.

    true_object_pose = torch.randn(32, 3 * parameters.num_object_control_points, 5, 13, 13).cuda()
    true_object_prob = torch.empty(32, 5, 13, 13, dtype=torch.long).random_(parameters.num_objects).cuda()
    object_mask = torch.zeros(5, 13, 13, dtype=torch.float32).cuda()
    object_mask[0, 0, 0] = 1.

    true = true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask

    print(model.total_loss(pred, true))
    
