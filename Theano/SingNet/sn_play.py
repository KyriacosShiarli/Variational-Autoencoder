from preprocess import pickle_loader,map_to_range_symmetric
import mido
import pygame
from VariationalAutoencoder import VA
import numpy as np

def play_from_encoder(directory):
    encoder = pickle_loader(directory + ".pkl")
    encoder.sample_rate = 44100
    sample_rate = encoder.sample_rate
    buffer_size = 20 # sample buffer for playback. Hevent really determined what i does qualitatively. Probably affects latency
    play_duration = 2000 # play duration in miliseconds
    pygame.mixer.pre_init(sample_rate, -16, 2,buffer = buffer_size) # 44.1kHz, 16-bit signed, stereo
    pygame.init()
    volLeft = 0.5;volRight=0.5 # Volume
    z_val = np.zeros([1,encoder.dimZ]) # Initial latent variable values
    port = mido.open_input(mido.get_input_names()[0]) # midi port. chooses the first midi device it detects.
    while True:
        mu_out = encoder.generateOutput(z_val)
        for msg in port.iter_pending():
            if msg.channel < z_val.shape[1]:
                z_val[0,msg.channel] = msg.value
            else:
                print "Midi channel beyond latent variables"
        mu_out = map_to_range_symmetric(mu_out,[-1,1],[-32768,32768])
        mu_out = mu_out.astype(np.int16)
        mu_out = np.array(zip(mu_out,mu_out)) # make stereo channel with same output
        sound = pygame.sndarray.make_sound(mu_out)
        channel = sound.play(-1)
        channel.set_volume(volLeft,volRight)
        pygame.time.delay(play_duration)
        sound.stop()

def play_from_encoder_no_midi(directory):
    # when no midi device is plugged in this function will play random
    # latent variable instances to inspect the capabilities of the network
    encoder = pickle_loader(directory + ".pkl")
    encoder.sample_rate = 44100
    sample_rate = encoder.sample_rate
    buffer_size = 1000
    play_duration = 2000 # play duration in miliseconds
    pygame.mixer.pre_init(sample_rate, -16, 2,buffer = buffer_size) # 44.1kHz, 16-bit signed, stereo
    pygame.init()
    volLeft = 0.5;volRight=0.5 # Volume
    z_val = np.zeros([1,encoder.dimZ]) # Initial latent variable values
    while True:
        print z_val
        mu_out = encoder.generateOutput(z_val)
        mu_out = map_to_range_symmetric(mu_out,[-1,1],[-32768,32768])
        mu_out = mu_out.astype(np.int16)
        mu_out = np.array(zip(mu_out,mu_out)) # make stereo channel with same output
        sound = pygame.sndarray.make_sound(mu_out)
        channel = sound.play(-1)
        channel.set_volume(volLeft,volRight)
        pygame.time.delay(play_duration)
        sound.stop()
        z_val = np.random.uniform(-30,30,[1,encoder.dimZ])


if __name__ == "__main__":
    play_from_encoder("models/encoder")


