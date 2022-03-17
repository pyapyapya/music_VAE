# music_VAE

### 목표
<b>Groove MIDI Dataset을 이용해 4마디에 해당하는 드럼 샘플 추출</b>


<hr/>


### 논문 요약
- Variational AutoEncoder(VAE)를 이용하면 의미있는 latent representation vector 생성 가능
- <b>Recurrent VAE 적용 </b>
  - RNN의 특성이 Latent Vector를 무시할 수 있음
  - Long-Term Sequential Data에서는 어려움이 있음
- <b>Hierarchical Decoder를 적용</b>
  - Posterior collapse를 피함
  - Long-Term sequence를 생성
- Sampling, Interpolation, Reconstruction에서 더 좋은 성능을 보여줌 (`Flat-Decoder`와 비교)


<hr/>


### 데이터 도메인 이해 및 설계 과정
- MusicVAE에서 전처리되어 벡터화 된 TFRecord를 사용하려 했으나, Pytorch로 구현하기 위해 처음부터 작업을 진행
- Groove MIDI DataSets에서 드럼 데이터의 클래스는 9개로 구성되어 있음. (경우의 수: 2^9=512)
- 데이터를 단순화 작업으로 time_signiture는 4/4로 고정하였고, velocity에 one-hot encoding을 적용
- 논문에서는 2-bar는 `Flat-Decoder` 를 사용하여, 4-bar에서도 `Flat-Decoder`우선 적용


<hr/>


### 전처리
- CSV 파일에서 MIDI 파일 정보 파싱
  - bpm, midi_filename, duration, split 데이터 사용

- MIDI 데이터 파싱 및 전처리
  - 음악이 0초부터 시작하는 `adjust_time` 구현
  - 16th note interval 만큼 `quantize` 기능 구현
  - MIDI 데이터를 벡터화 할 수 있는 `piano_roll` 구현 (batch, 9 channels, Max_Sequence)
- MIDI 학습 데이터 구성
  - 데이터를 4-bar 단위로 나누는 `Split_Sequence` 기능 구현

### 학습
<b>Encoder</b>
  - Input Data를 바탕으로 latent vector를 생성하는 역할
  - 학습 데이터를 Bi-LSTM에 적용하여 Latent Vector (mu, sigma) 추출
  - mu, sigma에 매개변수 조정 기법(reparameterization trick)을 이용하여 새로운 latent vector 생성 기능 구현

<b>구현 완료</b>


<hr/>

<b>Conductor</b>
- RNN based Embedding Vector를 생성하여 Decoder의 초기 값을 생성하는 역할
- latent를 loss함수로 `ELBO`를 사용하였고, 이는 `reconstruction accuracy` 와 `sampling quality`를 의미
- ELBO를 학습시키는데 두가지 방향이 있음
  - Beta-VAE & Free Bits
        - Beta-VAE는 ELBO의 두 항은 Trade-Off 인데, 이를 조절하는 역할
        - Free Bits는 KL-Divergence를 lower-bound(Free Bits)를 최소화하고, 나머지는 reconstruction에 학습
  - Latent Space Manipulation
        -  z1과 z2를 Interpolation하여 새로운 latent vector를 생성하여 semantically meaningful한 음악 생성
        -  이에 대한 결과로 `conductor vector` 생성



<b>Decoder</b>
  - Conductor에서 받은 latent 정보를 가지고 음악을 생성하는 역할
  - Long-Term 학습을 잘하기 위해 RNN based `Hierarchical Decoder` 사용
  - Decoder는 Conductor Vector와 concatnate되어 `Sub-Sequence Data`를 생성
