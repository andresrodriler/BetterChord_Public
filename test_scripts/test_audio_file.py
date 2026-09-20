import sys
from betterchord.audio_processing import load_audio, find_chord_position_time, create_spectrogram
import os

def test_audio_file(file_path):
    if not os.path.exists(file_path):
        print(f"\nError: File not found at '{file_path}'")
        return False

    try:
        print("\n Loading audio...")
        y, sr = load_audio(file_path)
        print(f" Size of loaded audio array: {len(y)} samples")
        print(f" Size of sr: {sr} Hz")

        spec = create_spectrogram(y, sr)
        
        # Show info about the loaded audio
        duration = len(y) / sr
        max_amplitude = abs(y).max()
        
        print(f"\n✓ Successfully loaded!")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Total samples: {len(y):,}")
        print(f"  Max amplitude: {max_amplitude:.3f}")
        print(f"  Spectrogram shape: {spec.shape}")

        if max_amplitude < 0.001:
            print("Did you record anything?")
        else:
            print("\n✓ Audio has good signal")


        return True
    except Exception as e:
        print(f"\nError loading audio: {e}")
        return False
    
    

if __name__ == "__main__":
    file_path = sys.argv[1]
    
    # Run the test
    success = test_audio_file(file_path)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)