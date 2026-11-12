from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document

from app.core.logger import get_logger

logger = get_logger()

DEFAULT_ROWS_PER_CHUNK = 35


class TabularExtractor:
    """
    Structured extractor for Tabular data (.csv and .xlsx).
    
    Capabilities:
    - Multi-sheet Excel extraction: Reads every worksheet, preserving sheet name & index.
    - Contextual chunking: Batches tabular rows with repeated column headers so
      each retrieved chunk retains column semantic context.
    - Row-level provenance: Annotates row_start, row_end, total_rows in chunk metadata.
    - Memory-bounded reading: Avoids loading entire giant workbooks into a single string.
    """

    def __init__(self, rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK):
        self.rows_per_chunk = rows_per_chunk

    def extract(self, file_path: Path) -> List[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            return self._extract_csv(file_path)
        elif suffix in [".xlsx", ".xls"]:
            return self._extract_excel(file_path)
        else:
            raise ValueError(f"Unsupported tabular file format: '{suffix}'")

    def _extract_csv(self, file_path: Path) -> List[Document]:
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)
        except Exception as e:
            logger.error(f"Failed to read CSV '{file_path.name}': {e}")
            raise ValueError(f"Malformed or unreadable CSV file: {e}")

        return self._dataframe_to_documents(
            df=df,
            source_name=file_path.name,
            sheet_name="Default",
            page_start=1,
            file_type=".csv",
        )

    def _extract_excel(self, file_path: Path) -> List[Document]:
        try:
            excel_file = pd.ExcelFile(file_path)
        except Exception as e:
            logger.error(f"Failed to open Excel workbook '{file_path.name}': {e}")
            raise ValueError(f"Malformed or unreadable Excel workbook: {e}")

        documents: List[Document] = []
        page_counter = 1

        try:
            for sheet_idx, sheet_name in enumerate(excel_file.sheet_names, start=1):
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                except Exception as e:
                    logger.warning(f"Could not read sheet '{sheet_name}' in '{file_path.name}': {e}")
                    continue

                sheet_docs = self._dataframe_to_documents(
                    df=df,
                    source_name=file_path.name,
                    sheet_name=sheet_name,
                    page_start=page_counter,
                    file_type=".xlsx",
                )
                documents.extend(sheet_docs)
                page_counter += max(1, len(sheet_docs))
        finally:
            excel_file.close()

        if not documents:
            documents.append(
                Document(
                    page_content="[Empty workbook: No readable sheets or data found]",
                    metadata={
                        "source": file_path.name,
                        "sheet_name": "None",
                        "page": 1,
                        "file_type": ".xlsx",
                    },
                )
            )

        logger.info(
            f"Extracted {len(documents)} tabular chunk(s) across "
            f"{len(excel_file.sheet_names)} sheet(s) from '{file_path.name}'"
        )
        return documents

    def _dataframe_to_documents(
        self,
        df: pd.DataFrame,
        source_name: str,
        sheet_name: str,
        page_start: int,
        file_type: str,
    ) -> List[Document]:
        if df.empty:
            return [
                Document(
                    page_content=f"[Empty Sheet: '{sheet_name}']",
                    metadata={
                        "source": source_name,
                        "sheet_name": sheet_name,
                        "page": page_start,
                        "total_rows": 0,
                        "file_type": file_type,
                    },
                )
            ]

        # Clean column names (strip whitespace and stringify)
        df.columns = [str(c).strip() for c in df.columns]

        # Fill NaNs with empty string for clean markdown display
        df_clean = df.fillna("")

        total_rows = len(df_clean)
        documents: List[Document] = []
        chunk_page = page_start

        for start_row in range(0, total_rows, self.rows_per_chunk):
            end_row = min(start_row + self.rows_per_chunk, total_rows)
            batch_df = df_clean.iloc[start_row:end_row]

            # Generate markdown table for this row slice
            table_md = batch_df.to_markdown(index=False)
            header_prefix = f"**Sheet:** `{sheet_name}` | **Rows:** {start_row + 1}–{end_row} of {total_rows}\n\n"
            content = header_prefix + table_md

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": source_name,
                        "sheet_name": sheet_name,
                        "page": chunk_page,
                        "row_start": start_row + 1,
                        "row_end": end_row,
                        "total_rows": total_rows,
                        "file_type": file_type,
                    },
                )
            )
            chunk_page += 1

        return documents
