import tensorflow as tf


hparams = tf.contrib.training.HParams(
    name="Tacotron-Wavenet-Vocoder",
    # tacotron hyper parameter
    cleaners='korean_cleaners',
    skip_path_filter=False,
    use_lws=False,
    
    # Audio
    sample_rate=24000,
    
    # shift can be specified by either hop_size(우선) or frame_shift_ms
    hop_size=300,
    fft_size=2048,
    win_size=1200,
    num_mels=80,

    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
    preemphasize=True,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    signal_normalization=True,
    allow_clipping_in_normalization=True,
    symmetric_mels=True,
    max_abs_value=4.,
        
    rescaling=True,
    rescaling_max=0.999, 
    
    trim_silence=True,
    trim_fft_size=512,
    trim_hop_size=128,
    trim_top_db=23,
    
    clip_mels_length=True,
    max_mel_frames=1000,


    l2_regularization_strength=0,
    sample_size=15000,
    silence_threshold=0,
    
    filter_width=2,
    gc_channels=32,
    
    input_type="raw",
    scalar_input=True,
    
    dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256,512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256,512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256,512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256,512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256,512],
    residual_channels=32,
    dilation_channels=32,
    quantization_channels=256,
    out_channels=30,
    skip_channels=512,
    use_biases=True,
    
    initial_filter_width=32,
    upsample_factor=[5, 5, 12],

    # wavenet training hp
    wavenet_batch_size=8,
    store_metadata=False,
    num_steps=200000,

    # Learning rate schedule
    wavenet_learning_rate=1e-3,
    wavenet_decay_rate=0.5,
    wavenet_decay_steps=300000,

    # Regularization parameters
    wavenet_clip_gradients=False,

    optimizer='adam',
    momentum=0.9,
    max_checkpoints=3,

    ####################################
    ####################################
    ####################################
    # TACOTRON HYPERPARAMETERS
    
    # Training
    adam_beta1=0.9,
    adam_beta2=0.999,
    use_fixed_test_inputs=False,
    
    tacotron_initial_learning_rate=1e-3,
    decay_learning_rate_mode=0,
    initial_data_greedy=True,
    initial_phase_step=8000,
    main_data_greedy_factor = 0,
    main_data=[''],
    prioritize_loss=False,

    # Model
    model_type='deepvoice',
    speaker_embedding_size=16,

    embedding_size=256,
    dropout_prob = 0.5,

    # Encoder
    enc_prenet_sizes=[256, 128],
    enc_bank_size=16,
    enc_bank_channel_size=128,
    enc_maxpool_width=2,
    enc_highway_depth=4,
    enc_rnn_size=128,
    enc_proj_sizes=[128, 128],
    enc_proj_width=3,

    # Attention
    attention_type='bah_mon_norm',
    attention_size=256,
    attention_state_size=256,

    # Decoder recurrent network
    dec_layer_num=2,
    dec_rnn_size=256,

    # Decoder
    dec_prenet_sizes=[256, 128],
    post_bank_size=8,
    post_bank_channel_size=128,
    post_maxpool_width=2,
    post_highway_depth=4,
    post_rnn_size=128,
    post_proj_sizes=[256, 80],
    post_proj_width=3,

    reduction_factor=5,

    # Eval
    min_tokens=10,
    min_iters=30,
    max_iters=200,
    skip_inadequate=False,

    griffin_lim_iters=60,
    power=1.5,

    recognition_loss_coeff=0.2,
    ignore_recognition_level=0
)

if hparams.use_lws:
    # Does not work if fft_size is not multiple of hop_size!!
    # sample size = 20480, hop_size=256=12.5ms. fft_size는 window_size를 결정하는데, 2048을 시간으로 환산하면 2048/20480 = 0.1초=100ms
    hparams.sample_rate = 20480
    hparams.hop_size = 256
    hparams.frame_shift_ms = None
    hparams.fft_size = 2048
    hparams.win_size = None
else:
    hparams.num_freq = int(hparams.fft_size/2 + 1)
    hparams.frame_shift_ms = hparams.hop_size * 1000.0/ hparams.sample_rate
    hparams.frame_length_ms = hparams.win_size * 1000.0/ hparams.sample_rate 


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
