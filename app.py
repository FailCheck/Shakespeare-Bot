import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------
# [1] 페이지 설정 (반드시 맨 처음에 와야 함)
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Shakespeare",
    page_icon="✒️",
    layout="wide"  # 화면을 넓게 씁니다
)

# ---------------------------------------------------------
# [2] 스타일 꾸미기 (CSS) - 글씨체나 박스 모양 예쁘게
# ---------------------------------------------------------
st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-size: 20px;
    }
    .main-text {
        font-family: 'Times New Roman', serif;
        font-size: 1.2rem;
        line-height: 1.6;
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# [3] AI 모델 클래스 (변경 없음)
# ---------------------------------------------------------
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.reshape(-1, out.shape[2])
        out = self.fc(out)
        return out

# ---------------------------------------------------------
# [4] 모델 로딩 함수 (캐시 사용)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load('shakespeare.pt', map_location=torch.device('cpu'))
    except FileNotFoundError:
        return None, None, None

    dic_size = checkpoint['dic_size']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']
    chars = checkpoint['chars']
    
    model = Net(dic_size, hidden_size, num_layers)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, chars, dic_size

model, chars, dic_size = load_model()

# ---------------------------------------------------------
# [5] 사이드바 (설정 메뉴)
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Shakespeare.jpg/220px-Shakespeare.jpg", width=150)
    st.title("⚙️ 설정 (Settings)")
    st.write("AI 작가의 성격을 조절하세요.")
    
    # 슬라이더들을 사이드바로 이동
    temperature = st.slider("🌡️ 창의성 (Temperature)", 0.1, 2.0, 0.8, help="낮으면 진지하고, 높으면 엉뚱해집니다.")
    length = st.slider("📏 글 길이 (Length)", 100, 1000, 300, step=100)
    
    st.divider()
    st.caption("Created by **Jay Jeon**")
    st.caption("Powered by PyTorch & LSTM")

# ---------------------------------------------------------
# [6] 메인 화면 구성
# ---------------------------------------------------------
st.title("✒️ AI Shakespeare Writer")
st.subheader("인공지능이 셰익스피어의 문체로 글을 이어 씁니다.")

# 화면을 왼쪽(입력)과 오른쪽(출력)으로 6:4 비율로 나눔
col1, col2 = st.columns([1, 1])

# --- 왼쪽: 입력란 ---
with col1:
    st.info("👇 첫 마디를 던져주세요.")
    user_input = st.text_input("입력 (영어):", "The king")
    
    if model is None:
        st.error("🚨 모델 파일(shakespeare.pt)이 없습니다!")
    
    generate_btn = st.button("✍️ 글쓰기 시작", type="primary", use_container_width=True)

    # 원리 설명 (포트폴리오용)
    with st.expander("ℹ️ 이 AI는 어떻게 작동하나요?"):
        st.markdown("""
        1. **데이터:** 셰익스피어 희곡 100만 자를 학습했습니다.
        2. **모델:** LSTM(Long Short-Term Memory) 신경망을 사용했습니다.
        3. **구조:** 2개의 층(Layers)을 쌓아 문맥을 깊이 이해합니다.
        4. **학습:** M-series GPU 가속을 통해 학습되었습니다.
        """)

# --- 오른쪽: 결과창 ---
with col2:
    if generate_btn and model is not None:
        char_dic = {c: i for i, c in enumerate(chars)}
        input_str = user_input
        generated_text = input_str
        
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            with torch.no_grad():
                for i in range(length):
                    x = [char_dic.get(c, 0) for c in input_str[-100:]] # 최근 100글자만 봄
                    
                    # One-hot encoding
                    x_one_hot = np.zeros((1, len(x), dic_size))
                    for t, char_idx in enumerate(x):
                        x_one_hot[0, t, int(char_idx)] = 1
                    
                    x_input = torch.tensor(x_one_hot, dtype=torch.float32)
                    output = model(x_input)
                    last_output = output[-1]
                    
                    prob = torch.softmax(last_output / temperature, dim=0).numpy()
                    char_index = np.random.choice(dic_size, p=prob)
                    
                    next_char = chars[char_index]
                    generated_text += next_char
                    input_str += next_char
                    
                    # 진행률 업데이트 (너무 빠르면 정신없으니 10글자마다 갱신)
                    if i % 10 == 0:
                        progress_bar.progress((i + 1) / length)
                        status_text.text(f"집필 중... {i+1}/{length}자")

            progress_bar.empty()
            status_text.empty()
            
            # 결과 예쁘게 보여주기
            st.success("작성 완료!")
            st.markdown(f'<div class="main-text">{generated_text}</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"에러 발생: {e}")