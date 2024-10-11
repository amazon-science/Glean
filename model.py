from utils.tools import *
from utils.contrastive import SupConLoss
        
class BertForModel(nn.Module):
    def __init__(self,model_name, num_labels, device=None):
        super(BertForModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.device = device
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.classifier = nn.Linear(768, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.backbone.to(self.device)
        self.classifier.to(self.device)

    def forward(self, X, output_hidden_states=False, output_attentions=False):
        """logits are not normalized by softmax in forward function"""
        outputs = self.backbone(**X, output_hidden_states=True)
        CLSEmbedding = outputs.hidden_states[-1][:,0]
        CLSEmbedding = self.dropout(CLSEmbedding)
        logits = self.classifier(CLSEmbedding)
        output_dir = {"logits": logits}
        if output_hidden_states:
            output_dir["hidden_states"] = outputs.hidden_states[-1][:, 0]
        if output_attentions:
            output_dir["attentions"] = outputs.attention
        return output_dir

    def mlmForward(self, X, Y):
        outputs = self.backbone(**X, labels=Y)
        return outputs.loss

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output
    
    def save_backbone(self, save_path):
        self.backbone.save_pretrained(save_path)


class CLBert(nn.Module):
    def __init__(self, args, model_name, device, num_labels, feat_dim=768, norm_classifier=True):
        super(CLBert, self).__init__()
        self.args = args
        self.model_name = model_name
        self.device = device
        self.num_labels = num_labels
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, feat_dim)
        )

        if args.architecture == 'Loop':
            print('\nUsing Loop Architecture')
            self.classifier = nn.Linear(feat_dim, self.num_labels)
        else: 
            print('\nUsing Default Architecture')
            self.classifier = nn.utils.weight_norm(nn.Linear(feat_dim, self.num_labels, bias=False))
            self.classifier.weight_g.data.fill_(1)
            if norm_classifier:
                self.classifier.weight_g.requires_grad = False        

        self.backbone.to(self.device)
        self.head.to(device)
        self.classifier.to(device)
        
    def forward(self, X, output_hidden_states=False, output_attentions=False, output_logits=False):
        """logits are not normalized by softmax in forward function"""
        outputs = self.backbone(**X, output_hidden_states=True, output_attentions=True)
        cls_embed = outputs.hidden_states[-1][:,0]
        features = F.normalize(self.head(cls_embed), dim=1)
        output_dir = {"features": features}
        
        if self.args.architecture in 'Loop':
            logits = self.classifier(self.head(cls_embed))
        else:
            x = nn.functional.normalize(cls_embed, dim=-1, p=2)
            logits = self.classifier(x)

        output_dir["logits"] = logits
        if output_hidden_states:
            output_dir["hidden_states"] = cls_embed
        if output_attentions:
            output_dir["attentions"] = outputs.attentions
        return output_dir

    def loss_cl(self, embds, label=None, mask=None, temperature=0.07, base_temperature=0.07):
        """compute contrastive loss"""
        loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)
        output = loss(embds, labels=label, mask=mask)
        return output
    
    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output

    def save_backbone(self, save_path):
        self.backbone.save_pretrained(save_path)


class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = 2
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(int(nepochs - warmup_teacher_temp_epochs)) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss