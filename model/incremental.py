import torch
import torch.nn as nn
import torch.nn.functional as F


class IncrementalBase(nn.Module):
    """
    RNN incremental processor.
    """
    def __init__(self, cfgs, token_size, label_size,
                 pretrained_emb=None):
        super(IncrementalBase, self).__init__()
        self.cfgs = cfgs
        self.dropout_1 = nn.Dropout(cfgs.DROPOUT_RNN)
        self.dropout_2 = nn.Dropout(cfgs.DROPOUT_RNN)
        self.proj = nn.Linear(cfgs.RNN_HIDDEN_SIZE, label_size)

        if cfgs.RNN_TYPE in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, cfgs.RNN_TYPE)(cfgs.WORD_EMBED_SIZE, 
                                                  cfgs.RNN_HIDDEN_SIZE, 
                                                  cfgs.RNN_LAYER,
                                                  batch_first=True,
                                                  dropout=cfgs.DROPOUT_RNN)
        else:
            raise KeyError('Model type not implemented!')

        # Xavier init
        for param in self.rnn.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x, states):
        # Assuming x is of shape (batch_size, 1) as nn.LSTM only provide final hidden states and memory cell, not the intermediate
        x = self.dropout_1(x)  # https://arxiv.org/abs/1708.02182 , Section 4.3 (Embedding dropout)

        # batch size is different for training and test time, not possible to initialize with batch size from yaml
        if states[0] is None and states[1] is None:

            h = torch.zeros(self.cfgs.RNN_LAYER, x.size(0), self.cfgs.RNN_HIDDEN_SIZE, dtype=torch.float, device=x.device)
            c = torch.zeros(self.cfgs.RNN_LAYER, x.size(0), self.cfgs.RNN_HIDDEN_SIZE, dtype=torch.float, device=x.device)
            states = (h, c)

        x, states = self.rnn(x, states)

        x = self.dropout_2(x)
        x = self.proj(x)

        return x, states


class LSTMN(nn.Module):
    """
    Controller module.
    """
    def __init__(self, cfgs, first_layer=True):
        super(LSTMN, self).__init__()
        self.cfgs = cfgs
        self.cell_history = []

        self.attn_sz = cfgs.CTRL_HIDDEN_SIZE
        self.h_summ = None
        self.c_summ = None

        if first_layer:
            self.W = nn.Linear(cfgs.CTRL_HIDDEN_SIZE + cfgs.WORD_EMBED_SIZE, 4 * cfgs.CTRL_HIDDEN_SIZE)
        else:
            self.W = nn.Linear(2 * cfgs.CTRL_HIDDEN_SIZE, 4 * cfgs.CTRL_HIDDEN_SIZE)

        # Attention weights
        self.v = nn.Linear(self.attn_sz, 1)
        self.linear_query = nn.Linear(cfgs.RNN_HIDDEN_SIZE, self.attn_sz)
        self.linear_key = nn.Linear(cfgs.CTRL_HIDDEN_SIZE, self.attn_sz)
        self.linear_summ = nn.Linear(cfgs.CTRL_HIDDEN_SIZE, self.attn_sz)

        # Xavier init
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def attn(self, query, phi_history, cell_history):
        # We assume phi_history is a list of phi with each shape of (batch_size, ctrl_hidden_dim). similar with cell_history

        # Shape (batch_size, curr_memory_size, ctrl_hidden_dim)
        phi_history = torch.stack(phi_history).transpose(0, 1)

        # Shape (batch_size, curr_memory_size, ctrl_hidden_dim)
        cell_history = torch.stack(cell_history).transpose(0, 1) 

        # Shape: (batch_size, curr_memory_size, hidden_dim)
        query = query.unsqueeze(1).repeat(1, phi_history.size(1), 1)

        # Shape: (batch_size, curr_memory_size, ctrl_hidden_dim)
        h_summ = self.h_summ.unsqueeze(1).repeat(1, phi_history.size(1), 1)

        # Shape: (batch_size, curr_memory_size, 1)
        scores = self.v(torch.tanh(self.linear_query(query) + self.linear_key(phi_history) + self.linear_summ(h_summ)))

        # Shape: (batch_size, curr_memory_size)
        attn_prob = torch.exp(F.log_softmax(scores.squeeze(-1), dim=-1))  # For numerical stability and valid probability.

        # Shape: (batch_size, ctrl_hidden_dim)
        h_summ = torch.bmm(
            phi_history.transpose(-2, -1), attn_prob.unsqueeze(-1)
        ).squeeze(-1)

        # Shape: (batch_size, ctrl_hidden_dim)
        c_summ = torch.bmm(
            cell_history.transpose(-2, -1), attn_prob.unsqueeze(-1)
        ).squeeze(-1)

        return h_summ, c_summ

    def forward(self, x, h_enc, phi_history):
        # phi history must be the same but cell history is different per layer.

        # batch size is different for training and test time, not possible to initialize with batch size from yaml
        if self.h_summ is None and self.c_summ is None:
            self.h_summ = torch.zeros(x.size(0), self.cfgs.CTRL_HIDDEN_SIZE, dtype=torch.float, device=x.device)
            self.c_summ = torch.zeros(x.size(0), self.cfgs.CTRL_HIDDEN_SIZE, dtype=torch.float, device=x.device)

        # Use only N last cache elements
        if len(phi_history) > 0:
            self.h_summ, self.c_summ = self.attn(
                h_enc, phi_history[-self.cfgs.CACHE_SIZE:],
                self.cell_history[-self.cfgs.CACHE_SIZE:]
            )

        # Shape: (batch_size, ctrl_hidden_dim + embed_size) for first layer, (batch_size, 2*ctrl_hidden_dim) otherwise
        concat_input = torch.cat((self.h_summ, x), 1)

        # Shape: (batch_size, 4 * ctrl_hidden_dim)
        gates = self.W(concat_input)

        # Shapes: (batch_size, ctrl_hidden_dim)
        f_t = torch.sigmoid(gates[:, :self.cfgs.CTRL_HIDDEN_SIZE])
        o_t = torch.sigmoid(gates[:, self.cfgs.CTRL_HIDDEN_SIZE: 2 * self.cfgs.CTRL_HIDDEN_SIZE])
        i_t = torch.sigmoid(gates[:, 2 * self.cfgs.CTRL_HIDDEN_SIZE:3 * self.cfgs.CTRL_HIDDEN_SIZE])
        ch_t = torch.tanh(gates[:, 3 * self.cfgs.CTRL_HIDDEN_SIZE:])

        c_ctrl = f_t * self.c_summ + i_t * ch_t
        h_ctrl = o_t * torch.tanh(c_ctrl)

        self.cell_history.append(c_ctrl)

        return h_ctrl, h_enc, phi_history


class RevisionModel(nn.Module):
    """
    Two-pass model for adaptive revision.
    """
    def __init__(self, cfgs, token_size, label_size,
                 reviser, pretrained_emb=None):
        super(RevisionModel, self).__init__()
        self.cfgs = cfgs
        self.encoder = IncrementalBase(cfgs, token_size, label_size, pretrained_emb)
        self.reviser = reviser
        self.ctrl_list = nn.ModuleList()

        for i in range(cfgs.CTRL_LAYER):
            first = True if i == 0 else False
            self.ctrl_list.append(LSTMN(cfgs, first_layer=first))

        # Initialize word embeddings
        self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.WORD_EMBED_SIZE
            )

        if cfgs.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        # Memory weights
        self.z_proj = nn.Linear(label_size, cfgs.RNN_HIDDEN_SIZE)
        self.phi_proj = nn.Linear(cfgs.RNN_HIDDEN_SIZE * 2, cfgs.CTRL_HIDDEN_SIZE)

        # Policy weights
        self.v_policy = nn.Linear(cfgs.CTRL_HIDDEN_SIZE, 1)
        self.threshold = cfgs.REV_THRESHOLD  # threshold for revision

        # Xavier init
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x, valid=False):  # Shape: (batch_size, max_len) for x
        hidden_history = []
        z_history = []
        phi_history = []
        output = []
        revision_output = []

        # For analysis of incremental metrics
        self.incr_out = []

        # Reset cell history for each LSTMN in nn.ModuleList()
        for layer in self.ctrl_list:
            setattr(layer, 'cell_history', [])

        # Reset state
        self.h_t = None
        self.c_t = None

        for layer in self.ctrl_list:
            setattr(layer, 'h_summ', None)
            setattr(layer, 'c_summ', None)

        max_len = x.size(-1)
        x_embed = self.embedding(x)  # Shape: (batch_size, max_len, embed_size)

        for t in range(max_len):
            x_t = x_embed[:, t:t+1, :]  # Shape (batch_size, 1, embed_size)

            pred, (self.h_t, self.c_t) = self.encoder(x_t, (self.h_t, self.c_t))  #Shape: (batch_size, 1, label_size), #Shape: (layers, batch_size, hidden_size)
            pred = pred.squeeze(1)
            self.incr_out.append(pred)  # for analysis only
            h_enc = self.h_t[-1]  # Shape: (batch_size, hidden_size) , use only from last layer

            x_ctrl = x_t.squeeze(1)

            for ctrl in self.ctrl_list:
                x_ctrl, h_enc, phi_history = ctrl(x_ctrl, h_enc, phi_history)

            # Shape: (batch_size, 1)
            policy = torch.sigmoid(self.v_policy(x_ctrl))

            if valid:
                # Remember for valid we assume batch_size = 1
                if policy.item() >= self.threshold:
                    # Hidden history must be appended with current h_enc first to update phi
                    hidden_history.append(h_enc)

                    # Revise
                    self.reviser.eval()
                    with torch.no_grad():
                        # Turn input to be a suitable format for transformer architecture -> original input before embedding
                        x_revise = x[:, :t+1]
                        revise_len = x_revise.size(1)
                        pad = torch.zeros(1, self.cfgs.MAX_TOKEN - revise_len, dtype=torch.long, device=x_revise.device)
                        x_revise = torch.cat((x_revise, pad), 1)
                        assert x_revise.size(1) == self.cfgs.MAX_TOKEN

                        # Get the logits from transformer
                        if self.cfgs.REVISER in ['linear-transformers-causal', 'transformers', 'linear-transformers']:
                            logits_revise = self.reviser(x_revise)  # Shape: (1, query_len, label_size)
                        elif self.cfgs.REVISER == 'incremental-transformers':
                            logits_revise = self.reviser(x_revise, valid=valid)  # Shape: (1, query_len, label_size)

                        mask = (x_revise != 0).squeeze()

                        logits_revise = logits_revise.squeeze(0)  # Shape: (query_len, label_size)
                        active_logits_revise = logits_revise[mask]  # Shape: (length of seq until time t, label_size)

                        # Update all z with new logits, also the history
                        z_update = torch.tanh(self.z_proj(active_logits_revise))  # Shape: (length of seq until time t, hidden_dim)
                        z_history = list(torch.split(z_update, 1))  # List of tensors with each of shape (1, hidden_dim)

                        # Update all phi with new z, also the history
                        h_enc_update = torch.stack(hidden_history).squeeze(1)  # Shape: (length of seq until time t, hidden_dim)
                        phi_update = torch.tanh(self.phi_proj(torch.cat((h_enc_update, z_update), 1)))  # Shape: (length of seq until time t, ctrl_hidden_dim)
                        phi_history = list(torch.split(phi_update, 1))

                        # Update all output with new output from trf.
                        output = list(torch.split(active_logits_revise, 1))

                    revision_output.append(policy)
                else:
                    z = torch.tanh(self.z_proj(pred))
                    phi = torch.tanh(self.phi_proj(torch.cat((h_enc, z), 1)))

                    z_history.append(z)
                    phi_history.append(phi)
                    hidden_history.append(h_enc)
                    output.append(pred)
                    revision_output.append(policy)
            else:
                # Training logic
                # Shape: (batch_size, hidden_dim)
                z = torch.tanh(self.z_proj(pred))

                # Shape: (batch_size, ctrl_hidden_dim)
                phi = torch.tanh(self.phi_proj(torch.cat((h_enc, z), 1)))

                z_history.append(z)
                phi_history.append(phi)
                hidden_history.append(h_enc)
                output.append(pred)
                revision_output.append(policy)

        # For analysis of incremental output
        self.incr_out = torch.stack(self.incr_out).transpose(0, 1)

        # Shape: (batch_size, max_len, label_size)
        output = torch.stack(output).transpose(0, 1)

        # Shape: (batch_size, max_len)
        revision_output = torch.stack(revision_output).transpose(0, 1).squeeze(-1)

        return output, revision_output
