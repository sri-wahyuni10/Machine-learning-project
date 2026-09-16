import numpy as np
import pandas as pd
import joblib
 
 
class RossmannSalesPredictor:
    """
    Wrapper inference untuk model forecasting Rossmann Sales.
 
    Cara pakai:
        predictor = RossmannSalesPredictor.load('model_bundle.pkl')
        pred = predictor.predict(store_id=1, target_date='2015-08-01', history_df=history)
 
    `history_df` minimal berisi 14 baris terakhir (urut tanggal naik) SEBELUM
    target_date, dengan kolom: ['Date', 'Sales', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday'].
    """
 
    MONTH_MAP = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
        5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
        9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
 
    FEATURE_ORDER = [
        'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
        'StoreType', 'Assortment', 'CompetitionDistance',
        'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
        'Promo2SinceWeek', 'Promo2SinceYear', 'competition_open_since_unknown',
        'competition_distance_missing', 'Month', 'is_promo2_active_month',
        'Year', 'competition_open_months', 'Sales_lag_1', 'Sales_lag_7',
        'Sales_lag_14', 'Sales_roll_mean_7', 'Sales_roll_std_7',
        'IsWeekend', 'WeekOfYear'
    ]
 
    def __init__(self, model, store_meta, categories):
        """
        model         : trained LightGBM model (best_model_rmspe)
        store_meta    : DataFrame hasil preprocessing store.csv (1 baris per Store,
                        sudah punya CompetitionDistance/CompetitionOpenSinceMonth/Year,
                        Promo2, Promo2SinceWeek/Year, PromoInterval, dan flag2 terkait)
        categories    : dict {kolom: list_kategori} persis seperti kategori X_train saat training
                        (Store, StateHoliday, StoreType, Assortment)
        """
        self.model = model
        self.store_meta = store_meta.set_index('Store')
        self.categories = categories
 
    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path):
        bundle = joblib.load(path)
        return cls(
            model=bundle['model'],
            store_meta=bundle['store_meta'],
            categories=bundle['categories']
        )
 
    def save(self, path):
        joblib.dump({
            'model': self.model,
            'store_meta': self.store_meta.reset_index(),
            'categories': self.categories
        }, path)
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_lag_rolling(self, history_df, target_date):
        """Hitung Sales_lag_1/7/14 dan Sales_roll_mean_7/std_7 dari histori.
 
        PENTING: history_df harus mencakup SEMUA hari kalender (termasuk hari
        Open==0 / toko tutup), persis seperti cara lag/rolling dihitung saat
        training. Kalau history_df sudah difilter Open==1 duluan, 14 baris
        terakhir itu sebenarnya mencakup LEBIH dari 14 hari kalender -> lag
        yang dihasilkan akan mismatch dengan training dan prediksi jadi bias.
        """
        history_df = history_df.sort_values('Date')
 
        # Validasi: pastikan rentang tanggal benar-benar 14 hari kalender berurutan
        # tepat sebelum target_date, bukan cuma "14 baris" apa adanya.
        expected_dates = pd.date_range(end=pd.Timestamp(target_date) - pd.Timedelta(days=1), periods=14)
        actual_dates = pd.DatetimeIndex(history_df['Date'].tail(14))
 
        if len(history_df) < 14 or not actual_dates.equals(pd.DatetimeIndex(expected_dates)):
            raise ValueError(
                "history_df tidak valid: harus berisi 14 hari KALENDER berurutan "
                f"({expected_dates.min().date()} s/d {expected_dates.max().date()}), "
                "TERMASUK hari-hari toko tutup (Open==0). Jangan filter Open==1 "
                "sebelum dikasih ke predictor - itu akan membuat lag/rolling "
                "mismatch dengan cara model dilatih."
            )
 
        sales = history_df['Sales']
 
        lag_1 = sales.iloc[-1]
        lag_7 = sales.iloc[-7]
        lag_14 = sales.iloc[-14]
        roll_mean_7 = sales.iloc[-7:].mean()
        roll_std_7 = sales.iloc[-7:].std()
 
        return lag_1, lag_7, lag_14, roll_mean_7, roll_std_7
 
    def _compute_promo2_active(self, promo_interval, target_date):
        if not promo_interval:
            return False
        month_abbr = self.MONTH_MAP[target_date.month]
        return month_abbr in promo_interval.split(',')
 
    def _compute_competition_months(self, comp_month, comp_year, target_date):
        return (target_date.year - comp_year) * 12 + (target_date.month - comp_month)
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, store_id, target_date, history_df, promo=0,
                state_holiday='0', school_holiday=0):
        """
        store_id      : ID toko (int)
        target_date   : tanggal yang mau diprediksi (str atau pd.Timestamp)
        history_df    : DataFrame histori penjualan toko ini, kolom minimal
                        ['Date', 'Sales'], urut tanggal, MINIMAL 14 baris SEBELUM target_date
        promo         : apakah target_date ada promo harian (0/1) - diketahui di muka
        state_holiday : kode hari libur di target_date ('0','a','b','c') - diketahui di muka
        school_holiday: apakah target_date libur sekolah (0/1) - diketahui di muka
        """
        target_date = pd.Timestamp(target_date)
 
        if store_id not in self.store_meta.index:
            raise ValueError(f"Store {store_id} tidak ditemukan di store_meta.")
        meta = self.store_meta.loc[store_id]
 
        # --- fitur dari histori (lag & rolling) ---
        lag_1, lag_7, lag_14, roll_mean_7, roll_std_7 = self._compute_lag_rolling(history_df, target_date)
 
        # --- fitur kalender ---
        day_of_week = target_date.isoweekday()  # 1=Senin ... 7=Minggu, konsisten dgn training
        month = target_date.month
        year = target_date.year
        week_of_year = int(target_date.isocalendar().week)
        is_weekend = int(day_of_week in [6, 7])
 
        # --- fitur promo2 & competition (dari store_meta) ---
        is_promo2_active = self._compute_promo2_active(meta['PromoInterval'], target_date)
        competition_open_months = self._compute_competition_months(
            meta['CompetitionOpenSinceMonth'], meta['CompetitionOpenSinceYear'], target_date
        )
 
        row = {
            'Store': store_id,
            'DayOfWeek': day_of_week,
            'Promo': promo,
            'StateHoliday': state_holiday,
            'SchoolHoliday': school_holiday,
            'StoreType': meta['StoreType'],
            'Assortment': meta['Assortment'],
            'CompetitionDistance': meta['CompetitionDistance'],
            'CompetitionOpenSinceMonth': meta['CompetitionOpenSinceMonth'],
            'CompetitionOpenSinceYear': meta['CompetitionOpenSinceYear'],
            'Promo2': meta['Promo2'],
            'Promo2SinceWeek': meta['Promo2SinceWeek'],
            'Promo2SinceYear': meta['Promo2SinceYear'],
            'competition_open_since_unknown': meta['competition_open_since_unknown'],
            'competition_distance_missing': meta['competition_distance_missing'],
            'Month': month,
            'is_promo2_active_month': is_promo2_active,
            'Year': year,
            'competition_open_months': competition_open_months,
            'Sales_lag_1': lag_1,
            'Sales_lag_7': lag_7,
            'Sales_lag_14': lag_14,
            'Sales_roll_mean_7': roll_mean_7,
            'Sales_roll_std_7': roll_std_7,
            'IsWeekend': is_weekend,
            'WeekOfYear': week_of_year,
        }
 
        X = pd.DataFrame([row])[self.FEATURE_ORDER]
 
        # --- pastikan kategori PERSIS sama dengan training (fix bug yg pernah kita temui) ---
        for col, cats in self.categories.items():
            X[col] = pd.Categorical(X[col], categories=cats)
 
        y_pred_log = self.model.predict(X)[0]
        y_pred = np.expm1(y_pred_log)
 
        return float(y_pred)
 
