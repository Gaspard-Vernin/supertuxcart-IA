import numpy as np
try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    # Fallback pour MoviePy 2.x
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# Génère 60 frames (2 secondes à 30 fps) de bruit aléatoire (matrices HxWxC)
frames = [np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8) for _ in range(60)]

print("Encodage de la vidéo en cours...")
clip = ImageSequenceClip(frames, fps=30)
# On force le codec standard pour être sûr de la compatibilité
clip.write_videofile("test_moviepy.mp4", codec="libx264") 
print("Vidéo générée avec succès !")