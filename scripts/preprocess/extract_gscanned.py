import os
import zipfile

def extract_ply_from_zips():
    cwd = os.getcwd()

    for file in os.listdir(cwd):
        if file.endswith(".zip"):
            zip_path = os.path.join(cwd, file)
            base_name = os.path.splitext(file)[0]  # e.g., CO3D_chair

            # optional: clean name (remove CO3D_ prefix if you want)
            folder_name = base_name

            output_dir = os.path.join(cwd, folder_name)
            os.makedirs(output_dir, exist_ok=True)

            print(f"\nProcessing: {file}")

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        # if member.lower().endswith(".ply"):
                        filename = os.path.basename(member)
                        if filename:  # skip directories
                            source = zip_ref.open(member)
                            os.makedirs(os.path.dirname(os.path.join(output_dir, member)), exist_ok=True)
                            target_path = os.path.join(output_dir, member)

                            with open(target_path, "wb") as target:
                                target.write(source.read())

                # delete zip after successful extraction
                # os.remove(zip_path)
                print(f"✔ Done: {file} → {output_dir}")

            except Exception as e:
                print(f"✖ Failed: {file} | Error: {e}")


if __name__ == "__main__":
    extract_ply_from_zips()