import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------
# 1. AI 모델 설계도 (학습 코드와 똑같아야 함!)
# ---------------------------------------------------------
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        # 층 개수(layers)를 변수로 받도록 수정!
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.reshape(-1, out.shape[2])
        out = self.fc(out)
        return out

# ---------------------------------------------------------
# 2. 웹사이트 화면 구성
# ---------------------------------------------------------
st.title("✒️ AI Shakespeare Writer (Pro)")
st.caption("100만 자의 셰익스피어 전집을 학습한 2층짜리 LSTM 모델입니다.")

# ---------------------------------------------------------
# 3. 뇌(.pt) 불러오기
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    # 1. 파일 읽기
    try:
        # map_location=torch.device('cpu')는 클라우드(CPU)에서 돌리기 필수!
        checkpoint = torch.load('shakespeare.pt', map_location=torch.device('cpu'))
    except FileNotFoundError:
        return None, None, None

    # 2. 저장된 설정값 가져오기
    # 학습할 때 저장했던 'save_data' 딕셔너리를 여기서 풉니다.
    dic_size = checkpoint['dic_size']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers'] # 2층이라는 정보를 여기서 가져옴!
    chars = checkpoint['chars']
    
    # 3. 모델 틀 만들기
    model = Net(dic_size, hidden_size, num_layers)
    
    # 4. 기억 심기 (가중치 로드)
    model.load_state_dict(checkpoint['model'])
    model.eval() # 평가 모드 (성적표 받을 준비)
    
    return model, chars, dic_size

model, chars, dic_size = load_model()

# ---------------------------------------------------------
# 4. 글쓰기 기능
# ---------------------------------------------------------
if model is None:
    st.error("🚨 'shakespeare.pt' 파일이 없습니다! Github에 업로드했는지 확인하세요.")
else:
    # 사용자 입력
    user_input = st.text_input("영어 단어를 입력하세요 (첫 마디를 던져주세요)", "The king")
    # [추가할 코드] 창의성 조절 슬라이더
    # 0.1 ~ 2.0 사이의 값을 조절. 기본값은 0.8
    temperature = st.slider("창의성 조절 (Temperature)", 0.1, 2.0, 0.8)

    if st.button("AI, 글을 써줘!"):
        # 글자 -> 숫자 사전
        char_dic = {c: i for i, c in enumerate(chars)}
        
        # 입력값 처리
        input_str = user_input
        if len(input_str) > 100: input_str = input_str[-100:] # 너무 길면 자름

        # 글쓰기 시작
        generated_text = input_str
        
        # 로딩바 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with torch.no_grad():
                for i in range(200): # 200글자 생성
                    # 현재 문장을 숫자로 변환
                    x = [char_dic.get(c, 0) for c in input_str] # 모르는 글자는 0번으로 처리
                    x = torch.tensor([x], dtype=torch.float32) # [1, len, vocab_size] (One-hot은 생략, 임베딩처럼 처리하거나 수정 필요하지만 일단 진행)
                    # 위 방식은 차원 에러 가능성 높음. 학습때 One-hot 했으므로 여기서도 해줘야 함.
                    
                    # One-hot Encoding (안전하게 다시 구현)
                    x_one_hot = np.zeros((1, len(input_str), dic_size))
                    for t, char_idx in enumerate(x[0]):
                        x_one_hot[0, t, int(char_idx)] = 1
                    
                    x_input = torch.tensor(x_one_hot, dtype=torch.float32)

                    # 예측
                    output = model(x_input)
                    
                    # 마지막 글자의 예측값 가져오기
                    last_output = output[-1]
                    
                    # 확률로 변환 (Softmax) 및 샘플링
                    prob = torch.softmax(last_output / temperature, dim=0).numpy()
                    
                    # 약간의 무작위성 추가 (Temperature) - 너무 뻔한 말만 안 하게
                    char_index = np.random.choice(dic_size, p=prob)
                    
                    # 숫자 -> 글자
                    next_char = chars[char_index]
                    generated_text += next_char
                    input_str += next_char # 다음 예측을 위해 붙임
                    
                    # 로딩바 업데이트
                    progress_bar.progress((i + 1) / 200)
                    status_text.text(f"집필 중... ({i+1}/200자)")

            st.success("작성 완료!")
            st.markdown(f"### 📜 AI의 창작물:\n> {generated_text}")
            
        except Exception as e:
            st.error(f"에러가 발생했습니다: {e}")