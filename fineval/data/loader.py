from abc import ABC, abstractmethod

import pandas as pd


class MarketDataset:
    """
    The Contract Dataclass

    Container for real and synthetic data at the loading stage.
    Holds raw prices in wide format(defined in 'prices' attribute) before preprocessing.

    Attributes:

         prices : pd.DataFrame
            Rows are timestamps, columns are ticker symbols, values are close
            prices. DatetimeIndex must be timezone-aware (UTC), columns must
            be strings, values must be float64. NaN where no trade occurred.

        name : str
            Human-readable label used in plots and reports (e.g. "Real", "AIL").

        is_synthetic : bool
            True for synthetic datasets, False for real market data.

        metadata : dict
            Optional. Generator hyperparameters, source paths, notes.
            Not read by the framework.

    Example::

                                    AAPL    MSFT    GOOG
            2019-09-03 09:30 UTC    205.1   137.8   1205.3
            2019-09-03 09:31 UTC    205.2   NaN     1205.5
            ...
    """

    def __init__(self, prices: pd.DataFrame, name: str, is_synthetic: bool, metadata: dict = None):
        self.prices = prices
        self.name = name
        self.is_synthetic = is_synthetic
        self.metadata = metadata if metadata is not None else {}


class BaseLoader(ABC):
    """
    Initial raw input for both synthetic and real data goes through
    this class.
    It returns a MarketDataset object that holds raw market data

    Attributes:

        name: str
            Label for Real or Synthetic market data (AIL, GBM, etc.)

        is_synthetic: bool
            Specifies the object is Real or Synthetic
    """

    def __init__(self, name: str, is_synthetic: bool):
        self.name = name
        self.is_synthetic = is_synthetic

    @abstractmethod
    def _load_raw(self) -> pd.DataFrame:
        """
        The logic of loading the dataset for framework

        Load raw data from disk and return a wide-format price DataFrame.

        Subclasses implement this with whatever format-specific logic
        their data requires (reading CSVs, pivoting parquet, etc.).

        Returns:
            pd.DataFrame, shape (T, N)
                T: number of timestamps for a single ticker symbol
                N: number of ticker symbols
        """
        pass

    def _validate(self, df: pd.DataFrame) -> None:
        """
        Checks the DataFrame against the MarketDataset contract.
        Raises ValueError with a helpful message on the first violation.

        Checks:
            - Is a pd.DataFrame
            - Index is pd.DatetimeIndex
            - Index is timezone-aware (UTC)
            - All column names are strings
            - DataFrame is non-empty
            - All values are numeric (float64)
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"[{self.name}] Expected pd.DataFrame, got {type(df).__name__}.")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"[{self.name}] Index must be pd.DatetimeIndex, got {type(df.index).__name__}."
            )

        if df.index.tz is None:
            raise ValueError(
                f"[{self.name}] Index must be timezone-aware (UTC). "
                "Hint: df.index = df.index.tz_localize('America/New_York')."
            )

        if not all(isinstance(c, str) for c in df.columns):
            raise ValueError(f"[{self.name}] All column names must be strings (ticker symbols).")

        if df.empty:
            raise ValueError(f"[{self.name}] DataFrame is empty.")

        non_numeric = [c for c in df.columns if not pd.api.types.is_float_dtype(df[c])]
        if non_numeric:
            raise ValueError(
                f"[{self.name}] Non-numeric columns: {non_numeric}. All columns must be float64."
            )

    def load(self) -> MarketDataset:
        """
        Orchestrates the loading workflow.

        Calls _load_raw(), validates the result against the MarketDataset
        contract, wraps it in a MarketDataset, and returns it.

        Returns:
            MarketDataset with validated prices.
        """
        prices = self._load_raw()
        self._validate(prices)
        return MarketDataset(
            prices=prices,
            name=self.name,
            is_synthetic=self.is_synthetic,
        )
