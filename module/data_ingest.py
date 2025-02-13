import pandas as pd  # Import library pandas untuk manipulasi data
import zipfile  # Import zipfile untuk ekstraksi file ZIP
import os  # Import os untuk operasi file system
import logging


import pandas as pd
import zipfile
import os


def ingest_file(file_path: str) -> pd.DataFrame:
    """Ingest data from file like (.txt, .csv, .json) to readable dataframe.

    Args:
        file_path (str): path of file data to ingest.

    Returns:
        pd.DataFrame
    """

    # Cek apakah file_path adalah file ZIP
    if file_path.endswith(".zip"):
        return ingest_zip_file(file_path)

    # Jika bukan ZIP, coba baca file berdasarkan ekstensi
    file_extension = os.path.splitext(file_path)[1].lower()

    # Menggunakan Pandas untuk membaca file sesuai dengan ekstensi
    if file_extension == ".csv":
        return pd.read_csv(file_path)
    elif file_extension == ".xlsx":
        return pd.read_excel(file_path)
    elif file_extension == ".json":
        return pd.read_json(file_path)
    elif file_extension == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. Supported formats: .csv, .xlsx, .json, .parquet."
        )


def ingest_zip_file(zip_file_path: str) -> pd.DataFrame:
    """Ingest a zip file to readable DataFrame.

    Args:
        zip_file_path (str): path of file .zip .

    ### Returns:
        pd.DataFrame
    """

    # Buka file ZIP dan ekstrak file yang ada di dalamnya
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        # Mendapatkan daftar semua file dalam ZIP
        zip_contents = zip_ref.namelist()

        if not zip_contents:
            raise ValueError("The ZIP archive is empty.")

        start_index = 0  # ubah jika perlu

        for idx in range(len(zip_contents)):
            # Mengambil file pertama yang ada di dalam ZIP jika itu bisa dibaca
            extracted_file = zip_contents[start_index + idx]

            # Ekstrak file
            zip_ref.extract(
                member=extracted_file,
                path=os.path.dirname(zip_file_path),
            )

        # Membaca file yang diekstrak sesuai dengan format
        return ingest_file(os.path.join(os.path.dirname(zip_file_path), extracted_file))


if __name__ == "__main__":
    pass
