# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""GRU cell"""
import mindspore as ms
from mindspore import nn, ops
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from src.weight_init import gru_default_state, gru_default_state_bw
from src.finetune_config import gru_cfg

class BidirectionGRU(nn.Cell):
    '''
    BidirectionGRU model

    Args:
        config: config of network
    '''
    def __init__(self, config, batch_size):
        super(BidirectionGRU, self).__init__()
        self.batch_size = batch_size
        self.embedding_size = config.encoder_embedding_size
        self.hidden_size = config.hidden_size
        self.weight_i, self.weight_h, self.bias_i, self.bias_h, self.init_h = gru_default_state(self.batch_size,
                                                                                                self.embedding_size,
                                                                                                self.hidden_size)

        # debug
        # para1 = gru_default_state(self.batch_size, self.embedding_size, self.hidden_size)
        # self.weight_i = para1[0]
        # self.weight_h = para1[1]
        # self.bias_i = para1[2]
        # self.bias_h = para1[3]
        # self.init_h = para1[4]

        self.weight_bw_i, self.weight_bw_h, self.bias_bw_i, self.bias_bw_h, self.init_bw_h = \
            gru_default_state_bw(self.batch_size, self.embedding_size, self.hidden_size)

        # debug
        # para2 = gru_default_state(self.batch_size, self.embedding_size, self.hidden_size)
        # self.weight_bw_i = para2[0]
        # self.weight_bw_h = para2[1]
        # self.bias_bw_i = para2[2]
        # self.bias_bw_h = para2[3]
        # self.init_bw_h = para2[4]

        # self.reverse = P.ReverseV2(axis=[1])
        self.reverse = P.ReverseV2(axis=[0])  # gil
        self.concat = P.Concat(axis=2)
        self.squeeze = P.Squeeze(axis=0)
        self.rnn = P.DynamicGRUV2()
        self.text_len = config.max_length
        self.cast = P.Cast()

    def construct(self, x):
        '''
        BidirectionGRU construction

        Args:
            x(Tensor): BidirectionGRU input

        Returns:
            output(Tensor): rnn output
            hidden(Tensor): hidden state
        '''
        x = self.cast(x, mstype.float16)
        y1, _, _, _, _, _ = self.rnn(x, self.weight_i, self.weight_h, self.bias_i, self.bias_h, None, self.init_h)
        # debug
        bw_x = self.reverse(x)
        y1_bw, _, _, _, _, _ = self.rnn(bw_x, self.weight_bw_i,
                                        self.weight_bw_h, self.bias_bw_i, self.bias_bw_h, None, self.init_bw_h)
        y1_bw = self.reverse(y1_bw)
        output1 = self.concat((y1, y1_bw))
        hidden = self.concat((y1[self.text_len-1:self.text_len:1, ::, ::],
                              y1_bw[self.text_len-1:self.text_len:1, ::, ::]))

        # output = self.concat((y1, y1))
        # hidden = self.concat((y1[self.text_len-1:self.text_len:1, ::, ::],
        #                       y1[self.text_len-1:self.text_len:1, ::, ::]))

        hidden = self.squeeze(hidden)
        return output1, hidden

class GRU(nn.Cell):
    '''
    GRU model

    Args:
        config: config of network
    '''
    def __init__(self, config, is_training=True):
        super(GRU, self).__init__()
        if is_training:
            self.batch_size = config.batch_size
        else:
            self.batch_size = config.eval_batch_size
        self.embedding_size = config.encoder_embedding_size
        self.hidden_size = config.hidden_size
        self.weight_i, self.weight_h, self.bias_i, self.bias_h, self.init_h = \
            gru_default_state(self.batch_size, self.embedding_size + self.hidden_size*2, self.hidden_size)
        self.rnn = P.DynamicGRUV2()
        self.cast = P.Cast()

    def construct(self, x):
        '''
        GRU construction

        Args:
            x(Tensor): GRU input

        Returns:
            output(Tensor): rnn output
            hidden(Tensor): hidden state
        '''
        x = self.cast(x, mstype.float16)
        y1, h1, _, _, _, _ = self.rnn(x, self.weight_i, self.weight_h, self.bias_i, self.bias_h, None, self.init_h)
        return y1, h1

if __name__ == '__main__':
    rnn1 = BidirectionGRU(gru_cfg, 4)
    ones = ops.Ones()
    input_tensor = ones((512, 4, 768), ms.float16)

    output = rnn1(input_tensor)
    print(output)
