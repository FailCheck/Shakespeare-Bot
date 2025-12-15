import torch
import torch.nn as nn
import numpy as np
import random

# ---------------------------------------------------------
# 1. 설정 (Config)
# ---------------------------------------------------------
# 수박을 한 번에 몇 조각씩 넣을 것인가? (64개씩 묶어서 학습)
BATCH_SIZE = 64
# 수박 한 조각의 길이 (글자 100개를 보고 다음 글자 맞추기)
SEQ_LENGTH = 100
# 학습 횟수 (반복 훈련) - 많이 할수록 똑똑해짐
NUM_EPOCHS = 2000 
# 모델의 층 개수 (더 깊게 쌓기)
HIDDEN_SIZE = 256
NUM_LAYERS = 2

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 사용 장치: {device}")

# ---------------------------------------------------------
# 2. 데이터 준비 (전처리)
# ---------------------------------------------------------
print("📚 데이터 읽는 중...")
try:
    with open('shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"✅ 전체 글자 수: {len(text)}자 (이제 다 먹일 수 있습니다!)")
except FileNotFoundError:
    print("❌ shakespeare.txt 파일을 찾을 수 없습니다!")
    exit()

# 글자 족보 만들기
chars = sorted(list(set(text)))
char_dic = {c: i for i, c in enumerate(chars)} # 글자 -> 숫자
dic_size = len(chars)

print(f"🔤 문자 종류: {dic_size}개")

# ---------------------------------------------------------
# 3. 배치를 만드는 국자 (Helper Function)
# ---------------------------------------------------------
# 이 함수가 핵심입니다! 전체 데이터에서 랜덤으로 64개 조각을 퍼옵니다.
def get_batch(text, batch_size, seq_length):
    input_batch = []
    target_batch = []
    
    for _ in range(batch_size):
        # 1. 랜덤한 위치를 하나 찍음
        start_idx = random.randint(0, len(text) - seq_length - 1)
        
        # 2. 그 위치부터 정해진 길이만큼 잘라냄
        chunk = text[start_idx : start_idx + seq_length + 1]
        
        # 3. 숫자로 변환
        encoded = [char_dic[c] for c in chunk]
        
        # 4. 문제(Input)와 정답(Target) 나누기
        # 문제: H e l l o (앞 5글자)
        # 정답: e l l o ! (뒤 5글자 - 한 칸 밀림)
        input_data = encoded[:-1]
        target_data = encoded[1:]
        
        input_batch.append(np.eye(dic_size)[input_data]) # One-hot Encoding
        target_batch.append(target_data)
        
    # 파이토치 텐서로 변환해서 GPU로 보냄
    inputs = torch.tensor(input_batch, dtype=torch.float32).to(device)
    targets = torch.tensor(target_batch, dtype=torch.long).to(device)
    
    return inputs, targets

# ---------------------------------------------------------
# 4. 모델 설계 (LSTM)
# ---------------------------------------------------------
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        # LSTM 결과는 3D인데, FC는 2D를 원함 -> 모양 맞추기
        out = out.reshape(-1, out.shape[2]) 
        out = self.fc(out)
        return out

model = Net(dic_size, HIDDEN_SIZE, NUM_LAYERS).to(device)

# 손실 함수와 최적화 도구
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# ---------------------------------------------------------
# 5. 학습 시작 (Training)
# ---------------------------------------------------------
print("\n🔥 스파르타 훈련 시작 (2000번 반복)...")

for epoch in range(NUM_EPOCHS):
    # 1. 국자로 데이터 퍼오기 (Batch)
    inputs, targets = get_batch(text, BATCH_SIZE, SEQ_LENGTH)
    
    # 2. 모델 예측
    outputs = model(inputs)
    
    # 3. 오차 계산 (정답이랑 얼마나 틀렸나?)
    # targets를 1줄로 쭉 펴야 함
    loss = criterion(outputs, targets.view(-1))
    
    # 4. 수정 (역전파)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 200번마다 성적표 출력
    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

# ---------------------------------------------------------
# 6. 저장 (Save)
# ---------------------------------------------------------
print("\n💾 똑똑해진 뇌 저장 중...")
save_data = {
    'model': model.state_dict(),
    'chars': chars,
    'hidden_size': HIDDEN_SIZE,
    'dic_size': dic_size,
    'num_layers': NUM_LAYERS # 층 개수도 저장해야 함
}
torch.save(save_data, 'shakespeare.pt')
print("✅ 저장 완료! (shakespeare.pt)")