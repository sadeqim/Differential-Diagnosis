#!/usr/bin/env python3
import argparse
import os
from glob import glob
import torch
import torchaudio
from transformers import AutoTokenizer, AutoProcessor, SeamlessM4Tv2ForSpeechToText

# Globals initialized in main()
processor = None
model = None
device = None

def process_folder(input_folder, output_folder):
    print(input_folder)
    os.makedirs(output_folder, exist_ok=True)
    wav_files = glob(os.path.join(input_folder, "*.wav"))
    for wav_file in wav_files:
        waveform, sr = torchaudio.load(wav_file)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)

        # Persian transcription
        inputs_pes = processor(audios=waveform, src_lang="pes", sampling_rate=16000, return_tensors="pt").to(device)
        output_tokens_pes = model.generate(**inputs_pes, tgt_lang="pes", num_beams=5)
        text_pes = processor.decode(output_tokens_pes[0], skip_special_tokens=True)

        # English (per your original logic)
        inputs_eng = processor(audios=waveform, src_lang="eng", sampling_rate=16000, return_tensors="pt").to(device)
        output_tokens_eng = model.generate(**inputs_eng, tgt_lang="eng", num_beams=5)
        text_eng = processor.decode(output_tokens_eng[0], skip_special_tokens=True)

        base_name = os.path.basename(wav_file).replace(".wav", "")
        pes_txt_file = os.path.join(output_folder, f"{base_name}_pes.txt")
        with open(pes_txt_file, "w") as f:
            f.write(f"Persian Transcription:\n{text_pes}\n")

        eng_txt_file = os.path.join(output_folder, f"{base_name}_eng.txt")
        with open(eng_txt_file, "w") as f:
            f.write(f"English Translation:\n{text_eng}\n")

        print(f"Processed {wav_file} -> Persian: {pes_txt_file}, English: {eng_txt_file}")

def main():
    global processor, model, device

    parser = argparse.ArgumentParser(description="Batch STT with SeamlessM4T v2")
    parser.add_argument("input_folder", help="Folder containing .wav files")
    parser.add_argument("output_folder", help="Folder to write transcripts")
    args = parser.parse_args()

    # From your first module (kept intact)
    tokenizer = AutoTokenizer.from_pretrained("facebook/seamless-m4t-v2-large", use_fast=False)
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large", tokenizer=tokenizer)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large")
    print("Model downloaded and cached.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Process the single input/output pair from CLI
    process_folder(args.input_folder, args.output_folder)

    # (Keeping your original parting print line)
    print("Download them manually from your server (e.g., via SCP).")

    # --- Your original zip/cleanup block was commented out; leaving it commented here too ---
    # import shutil
    # zip_paths = []
    # output_folder = args.output_folder
    # zip_name = output_folder.replace("/", "_")
    # shutil.make_archive(zip_name, 'zip', output_folder)
    # zip_paths.append(f"{zip_name}.zip")
    # print(f"Zipped {output_folder} to {zip_name}.zip")
    # shutil.rmtree(output_folder)
    # print(f"Removed {output_folder}")
    # print("All processing complete. Zips are at:", zip_paths)

if __name__ == "__main__":
    main()


# python main.py /home/user01/facebook-denoiser/results /home/user01/facebook-seamless/results