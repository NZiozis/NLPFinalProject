import torch
import torch.nn as nn

class SentenceDecoder(nn.Module):
    def __init__(self, args, max_seq_length=30):
        """Set the hyper-parameters and build the layers."""
        super(SentenceDecoder, self).__init__()
        self.sentDec_inDim = args.sentDec_inDim
        self.word_dim = args.word_dim
        self.sentDec_hiddens = args.sentDec_hiddens
        self.vocab_len = args.vocab_len
        self.sentDec_nlayers = args.sentDec_nlayers

        self.lstm = nn.LSTM(self.word_dim, self.sentDec_hiddens, self.sentDec_nlayers, batch_first=True)
        self.linear = nn.Linear(self.sentDec_hiddens, self.vocab_len)
        self.max_seg_length = max_seq_length

        self.linear_project = nn.Linear(self.sentDec_inDim, self.word_dim)

    def forward(self, recipe_enc, sent_lens):
        """Decode sentence feature vectors and generates sentences."""
        # recipe_enc --> [Nb, 1024]
        # word_embs  --> [Nb, Ns, 256]
        # len(sent_lens)  --> Nb

        features = self.linear_project(recipe_enc)  # [Nb, 256]
        features = features.unsqueeze(0)
        #word_embs = torch.cat((features.unsqueeze(1), word_embs), 1)  # torch.Size([Nb, Ns + 1, 256])
        #packed = pack_padded_sequence(word_embs, sent_lens, batch_first=True)
        #packed = pack_padded_sequence(features, sent_lens, batch_first=True)
        # [0] -> [sum(sent_lens), 256]   [1] -> [sent_lens[0]]

        #out, _ = self.lstm(packed)  # [0] -> [sum(sent_lens), 512]   [1] -> [sent_lens[0]]

        out, _ = self.lstm(features)
        outputs = self.linear(out[0])  # [sum(sent_lens), Nw] -- Nw = number of words in the vocabulary

        return outputs

    def forward_sample(self, embed_words, inputPred, hidden):

        embedded = embed_words(inputPred).unsqueeze(1)  # [Nb, 256]
        output, hidden = self.lstm(embedded, hidden)

        output = self.linear(output.squeeze(0))
        output = F.log_softmax(output, dim=1)
        return output, hidden

    def sample(self, recipe_enc, embed_words, states=None):
        # recipe_enc --> [Nb, 1024]
        max_seq_length = 20
        sampled_ids = []
        features = self.linear_project(recipe_enc)  # [Nb, 256]
        inputs = features.unsqueeze(1)  # [Nb, 1, 256]
        for i in range(max_seq_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: [Nb, 1, 512]

            outputs = self.linear(hiddens.squeeze(1))  # [Nb, Nw] -- Nw = number of words in the vocabulary

            _, predicted = outputs.max(1)  # Nb
            sampled_ids.append(predicted)

            inputs = embed_words(predicted)  # [Nb, 256]
            inputs = inputs.unsqueeze(1)  # [Nb, 1,256]

        sampled_ids = torch.stack(sampled_ids, 1)  # [Nb, max_seq_length]

        return sampled_ids

    def sample_greedy_decode(self,   step_enc, embed_words, decoder_hidden=None):
        '''
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :return: decoded_batch
        '''
        batch_size = 1
        MAX_LENGTH = 20
        decoded_batch = torch.zeros((batch_size, MAX_LENGTH))

        features = self.linear_project(step_enc)  # [Nb, 256]
        decoder_input = features.unsqueeze(1)  # [Nb, 1, 256]
        decoder_output, decoder_hidden = self.lstm(decoder_input, decoder_hidden)  # hiddens: [Nb, 1, 512]
        outputs = self.linear(decoder_output.squeeze(1))  # [Nb, Nw] -- Nw = number of words in the vocabulary
        _, predicted = outputs.max(1)  # Nb
        decoded_batch[:, 0] = predicted

        decoder_input = torch.LongTensor([predicted for _ in range(batch_size)])

        for t in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_sample(embed_words, decoder_input.cuda(), decoder_hidden)

            topv, topi = decoder_output.data.topk(1)  # get candidates
            topi = topi.view(-1)
            decoded_batch[:, t] = topi

            decoder_input = topi.detach()

        return decoded_batch

    def sample_beam_decode(self,  step_enc, embed_words, decoder_hidden=None):
        '''
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :return: decoded_batch
        vocab('<start>') = 30168
        vocab('<end>') = 30169
        '''
        batch_size = 1
        MAX_LENGTH = 20
        EOS_token = 30169
        beam_width = 5
        topk = 10  # how many sentence do you want to generate

        features = self.linear_project(step_enc)  # [Nb, 256]
        decoder_input = features.unsqueeze(1)  # [Nb, 1, 256]
        decoder_output, decoder_hidden = self.lstm(decoder_input, decoder_hidden)  # hiddens: [Nb, 1, 512]
        outputs = self.linear(decoder_output.squeeze(1))  # [Nb, Nw] -- Nw = number of words in the vocabulary
        _, predicted = outputs.max(1)  # Nb

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([predicted])

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, nds = nodes.get()
            decoder_input = torch.LongTensor([nds.wordid])
            decoder_hidden = nds.h

            if nds.wordid.item() == EOS_token and nds.prevNode != None:
                endnodes.append((score, nds))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden = self.forward_sample(embed_words, decoder_input.cuda(), decoder_hidden)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k]
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, nds, decoded_t, nds.logp + log_p, nds.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        decoded_batch_all = []
        for score, nco in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(nco.wordid)
            # back trace
            while nco.prevNode != None:
                nco = nco.prevNode
                utterance.append(nco.wordid)

            utterance = utterance[::-1]
            decoded_batch_all.append(utterance)

        decoded_batch = torch.zeros((batch_size, min(MAX_LENGTH, len(decoded_batch_all[0]))))
        for klik in range(min(MAX_LENGTH, len(decoded_batch_all[0]))):
            decoded_batch[:, klik] = decoded_batch_all[0][klik].cuda()

        return decoded_batch