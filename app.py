import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Jay's Baby GPT", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main-text {
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        line-height: 1.6;
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. GPT 모델 구조 (학습 코드와 똑같이 복사해야 함)
# ---------------------------------------------------------
# 설정값 (학습할 때 쓴 것과 똑같이 맞춰야 함)
BLOCK_SIZE = 64
N_EMBD = 128
N_HEAD = 4
N_LAYER = 2
DROPOUT = 0.2
VOCAB_SIZE = 65 # 셰익스피어 데이터 문자 개수 (대략 65개)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device='cpu')) # CPU로 강제
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# ---------------------------------------------------------
# 3. 모델 로딩 (GPT 전용)
# ---------------------------------------------------------
@st.cache_resource
def load_gpt_model():
    # 1. 깡통 모델 만들기
    model = GPTLanguageModel()
    
    # 2. 학습된 가중치(기억) 불러오기
    try:
        # map_location='cpu' 필수 (클라우드는 GPU가 없을 수 있음)
        state_dict = torch.load('baby_gpt.pt', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        return None, str(e)

    # 3. 문자 족보(Vocab) 만들기 (학습 때 쓴 것과 똑같아야 함)
    # 셰익스피어 데이터에 있는 모든 글자 (총 65개)
    chars = sorted(list(set("\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")))
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    return model, stoi, itos

model, stoi, itos = load_gpt_model()

# ---------------------------------------------------------
# 4. 화면 구성
# ---------------------------------------------------------
with st.sidebar:
    st.title("🤖 Jay's Baby GPT")
    st.caption("Transformer Architecture (2017) 구현체")
    st.markdown("---")
    temperature = st.slider("창의성 (Temperature)", 0.5, 1.5, 0.8)
    max_tokens = st.slider("생성 길이", 100, 1000, 300)
    st.info("이 모델은 문맥을 파악하는 'Attention' 메커니즘을 사용합니다.")

st.title("🧠 Baby GPT: The Beginning")
st.write("LSTM(순차 처리)을 넘어, **Transformer(병렬 처리)** 시대로 오신 것을 환영합니다.")

col1, col2 = st.columns([1, 1])

with col1:
    start_str = st.text_input("첫 문장을 입력하세요:", "The meaning of life is")
    btn = st.button("GPT, 생각해서 글을 써줘!", type="primary")

with col2:
    if btn:
        if isinstance(model, str): # 에러 메시지인 경우
            st.error(f"모델 로딩 실패: {model}\n'baby_gpt.pt' 파일을 업로드했는지 확인하세요.")
        else:
            status = st.empty()
            progress = st.progress(0)
            
            # 초기 입력값 숫자로 변환
            context = [stoi.get(c, 0) for c in start_str]
            idx = torch.tensor([context], dtype=torch.long)
            
            generated_text = start_str
            
            with torch.no_grad():
                for i in range(max_tokens):
                    # Context Window 자르기 (최근 64글자만 봄)
                    idx_cond = idx[:, -BLOCK_SIZE:]
                    
                    # 예측
                    logits = model(idx_cond)
                    logits = logits[:, -1, :] # 마지막 글자만
                    
                    # 확률 조작 (Temperature)
                    probs = F.softmax(logits / temperature, dim=-1)
                    
                    # 다음 글자 뽑기
                    idx_next = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat((idx, idx_next), dim=1)
                    
                    # 결과 누적
                    next_char = itos[idx_next.item()]
                    generated_text += next_char
                    
                    # 로딩바
                    if i % 10 == 0:
                        status.text(f"GPT가 생각 중... ({i}/{max_tokens})")
                        progress.progress((i+1)/max_tokens)
            
            status.empty()
            progress.empty()
            st.success("생성 완료!")
            st.markdown(f'<div class="main-text">{generated_text}</div>', unsafe_allow_html=True)