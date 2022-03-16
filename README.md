# music_VAE

### 목표
<b>Groove MIDI Dataset을 이용해 4마디에 해당하는 드럼 샘플 추출</b>

### 논문 요약
- Variational AutoEncoder(VAE)를 이용하면 의미있는 latent representation vector 생성 가능
- Long-Term Sequential Data에서는 어려움이 있음
- <b>Recurrent VAE 적용 </b>
  - Latent Vector가 Global한 정보를 잘 capture함
- <b>Hierarchical Decoder를 적용</b>
  - Posterior collapse를 피함
  - Long-Term sequence를 생성
- Sampling, Interpolation, Reconstruction에서 더 좋은 성능을 보여줌 (Flat-Decoder와 비교)

### 데이터 도메인 이해 및 설계 과정
- MusicVAE에서 전처리되어 벡터화 된 TFRecord를 사용하려 했으나, Pytorch로 구현하기 위해 처음부터 작업을 진행
- Groove MIDI DataSets에서 드럼 데이터의 클래스는 9개로 구성되어 있음. (경우의 수: 2^9=512)
- 데이터를 단순화 작업으로 time_signiture는 4/4로 고정하였고, velocity에 one-hot encoding을 적용
- 논문에서는 2-bar는 `Flat-Decoder` 를 사용하여, 4-bar에서도 `Flat-Decoder`우선 적용

### 전처리
- CSV 파일에서 MIDI 파일 정보 파싱
- MIDI 데이터 파싱 및 전처리
  - 음악이 0초부터 시작하는 `adjust_time` 구현
  - 16th note interval 만큼 `quantize` 기능 구현
  - MIDI 데이터를 벡터화 할 수 있는 `piano_roll` 구현 (9 channels, Max_Sequence)
- MIDI 학습 데이터 구성
  - 음악을 데이터를 4-bar 단위로 나누는 기능 구현

### 학습
- Encoder
  - 학습 데이터를 Bi-LSTM에 적용하여 Latent Vector (mu, sigma) 추출

- Conductor 

- Decoder

### 생성
- Reconstruction
- Interpolation
- Attribute Vector
