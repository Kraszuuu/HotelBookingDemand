from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


class Plotter:
    def __init__(self, seed: int = 268555):
        self.seed = seed

    # ── ADR seasonality (cell 34) ───────────────────────────────────

    def plot_adr_seasonality(self, data: pd.DataFrame) -> None:
        df = data.copy()
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
        df = df.dropna(subset=["hotel", "arrival_date", "adr"]).copy()
        hotels = sorted(df["hotel"].unique())
        if len(hotels) != 2:
            raise ValueError(f"Expected 2 hotels, got {len(hotels)}")
        fig, axes = plt.subplots(2, 2, figsize=(18, 8), sharey=False)
        fig.suptitle("Mean and Median ADR: weekly interval and week number", fontsize=14, fontweight="bold")
        for ci, hotel in enumerate(hotels):
            hd = df[df["hotel"] == hotel].copy()
            self._plot_adr_weekly_date(axes[0, ci], hotel, hd)
            self._plot_adr_week_number(axes[1, ci], hotel, hd)
        fig.tight_layout()
        plt.show()

    def _plot_adr_weekly_date(self, ax: Any, hotel: str, hotel_df: pd.DataFrame) -> None:
        w = hotel_df.set_index("arrival_date")["adr"].resample("W-MON").agg(adr_mean="mean", adr_median="median").reset_index().rename(columns={"arrival_date": "week_start"})
        wk = w["week_start"].dt.isocalendar().week.astype(int)
        w["is_summer"] = wk.between(27, 35)
        starts = pd.to_datetime(w["week_start"])
        ends = starts + pd.Timedelta(days=6)
        win = []
        for s, e in zip(starts, ends):
            y = int(s.year)
            win.append((s <= pd.Timestamp(year=y, month=1, day=6)) and (e >= pd.Timestamp(year=y - 1, month=12, day=24)) or
                       (s <= pd.Timestamp(year=y + 1, month=1, day=6)) and (e >= pd.Timestamp(year=y, month=12, day=24)))
        w["is_winter"] = win
        for fc, cl, lb in [("is_summer", "#1f7a1f", "Summer holiday"), ("is_winter", "#b22222", "Winter holiday")]:
            md = pd.DataFrame({"date": pd.to_datetime(w["week_start"]), "flag": w[fc]}).sort_values("date")
            md["gid"] = (md["flag"] != md["flag"].shift(1, fill_value=False)).cumsum()
            first = True
            for _, p in md[md["flag"]].groupby("gid"):
                s = p["date"].iloc[0]; e = p["date"].iloc[-1] + pd.Timedelta(days=7)
                ax.axvspan(s, e, color=cl, alpha=0.25, label=lb if first else None)
                first = False
        ax.plot(w["week_start"], w["adr_mean"], color="#b22222", linewidth=1.2, label="Mean ADR")
        ax.plot(w["week_start"], w["adr_median"], color="#228b22", linewidth=1.2, label="Median ADR")
        ax.set_title(f"{hotel} - weekly interval (date)");
        ax.set_xlabel("Week start"); ax.set_ylabel("ADR"); ax.legend()

    def _plot_adr_week_number(self, ax: Any, hotel: str, hotel_df: pd.DataFrame) -> None:
        w = hotel_df[["arrival_date_week_number", "adr"]].dropna().groupby("arrival_date_week_number")["adr"].agg(adr_mean="mean", adr_median="median").sort_index().reset_index()
        ax.plot(w["arrival_date_week_number"], w["adr_mean"], color="#b22222", linewidth=1.2, label="Mean ADR")
        ax.plot(w["arrival_date_week_number"], w["adr_median"], color="#228b22", linewidth=1.2, label="Median ADR")
        ax.axvspan(27, 35, color="#1f7a1f", alpha=0.25, label="Summer holiday")
        ax.axvspan(51.5, 53.5, color="#b22222", alpha=0.25, label="Winter holiday")
        ax.axvspan(0.5, 1.5, color="#b22222", alpha=0.25)
        ax.set_title(f"{hotel} - by week number");
        ax.set_xlabel("Week of Year"); ax.set_ylabel("ADR")
        ax.set_xlim(1, 53); ax.set_xticks(np.arange(1, 54, 4)); ax.legend()

    # ── Country distribution (cell 43) ──────────────────────────────

    def plot_country_distribution(self, data: pd.DataFrame) -> None:
        hotels = sorted(data["hotel"].dropna().unique())
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Country Distribution (Top 10 + Others) by Hotel Type", fontsize=14, fontweight="bold")
        for idx, hotel in enumerate(hotels):
            cc = data[data["hotel"] == hotel]["country"].value_counts(dropna=False)
            if cc.empty:
                continue
            ps = cc.head(10).copy()
            other = cc.iloc[10:].sum()
            if other > 0:
                ps["Others"] = other
            n = len(ps)
            base = "#b22222" if idx == 0 else "#1f7a1f"
            rgb = mcolors.to_rgb(base)
            colors = [(*rgb, max(0.3, 1.0 - i * 0.7 / max(n - 1, 1))) for i in range(n)]
            if "Others" in ps.index:
                colors[list(ps.index).index("Others")] = (0.8, 0.8, 0.8, 1.0)
            explode = [0.04] * n
            if "Others" in ps.index:
                explode[list(ps.index).index("Others")] = 0.1
            def fmt(pct): return f"{pct:.1f}%" if pct >= 4 else ""
            axes[idx].pie(ps.values, labels=ps.index.astype(str), autopct=fmt,
                          startangle=90, colors=colors, explode=explode,
                          wedgeprops={"edgecolor": "#000000", "linewidth": 1.0})
            axes[idx].set_title(hotel)
        fig.tight_layout(); plt.show()

    # ── Top countries by ADR (cell 46) ──────────────────────────────

    def plot_top_countries_by_adr(self, data: pd.DataFrame, top_n: int = 10, use_median: bool = False) -> None:
        df = data[(data["reservation_status"] == "Check-Out") & data["adr"].notna() & data["country"].notna()].copy()
        top = df["country"].value_counts().head(top_n).index.tolist()
        df = df[df["country"].isin(top)]
        agg = df.groupby("country")["adr"].agg(["mean", "median", "count"]).reset_index()
        col = "median" if use_median else "mean"
        agg = agg.sort_values(col, ascending=False)
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#b22222" if c == "PRT" else "#2e8b57" for c in agg["country"]]
        bars = ax.bar(range(len(agg)), agg[col], color=colors)
        ax.set_xticks(range(len(agg))); ax.set_xticklabels(agg["country"], rotation=45, ha="right")
        for i, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width() / 2., b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
        metric = "Median" if use_median else "Mean"
        ax.set_title(f"Top {top_n} Countries by {metric} ADR", fontsize=14, fontweight="bold")
        ax.set_xlabel("Country"); ax.set_ylabel(f"{metric} ADR (€)")
        for i, row in enumerate(agg.itertuples()):
            ax.text(i, 0.02, f"n={int(row.count)}", ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
        fig.tight_layout(); plt.show()

    # ── Children occupancy (cell 49) ────────────────────────────────

    def plot_children_occupancy(self, data: pd.DataFrame) -> None:
        occ = self._prepare_children_base(data)
        exp = self._expand_children_daily(occ)
        hotels = sorted(exp["hotel"].dropna().unique())
        if len(hotels) != 2:
            raise ValueError(f"Expected 2 hotels, got {len(hotels)}")
        fig, axes = plt.subplots(2, 2, figsize=(18, 8))
        fig.suptitle("Occupancy: guests without children vs with children", fontsize=14, fontweight="bold")
        for ci, hotel in enumerate(hotels):
            hd = exp[exp["hotel"] == hotel].copy()
            self._plot_children_weekly(axes[0, ci], hotel, hd)
            self._plot_children_arrival_week(axes[1, ci], hotel, hd)
        fig.tight_layout(); plt.show()

    def _prepare_children_base(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
        df["total_nights"] = (df["stays_in_week_nights"].fillna(0) + df["stays_in_weekend_nights"].fillna(0)).astype(int)
        df["children"] = df["children"].fillna(0); df["babies"] = df["babies"].fillna(0)
        df["guestGroup"] = np.where((df["children"] + df["babies"]) > 0, "With children", "Without children")
        occ = df[["hotel", "arrival_date", "arrival_date_week_number", "total_nights", "guestGroup"]].dropna(
            subset=["hotel", "arrival_date", "arrival_date_week_number"]).copy()
        bt = occ["guestGroup"].value_counts().to_frame(name="count")
        bt["percentage"] = (bt["count"] / bt["count"].sum() * 100).round(2)
        adr_stats = data.assign(children=data["children"].fillna(0), babies=data["babies"].fillna(0)).assign(
            guestGroup=lambda d: np.where((d["children"] + d["babies"]) > 0, "With children", "Without children")
        ).dropna(subset=["hotel", "arrival_date", "arrival_date_week_number", "adr"]).groupby("guestGroup")["adr"].agg(
            adrMean="mean", adrMedian="median", adrMin="min", adrMax="max").round(2)
        print(bt.join(adr_stats, how="left"))
        return occ[occ["total_nights"] > 0].copy()

    def _expand_children_daily(self, occ: pd.DataFrame) -> pd.DataFrame:
        exp = occ.loc[occ.index.repeat(occ["total_nights"])].copy()
        exp["day_offset"] = exp.groupby(level=0).cumcount()
        exp["stay_date"] = exp["arrival_date"] + pd.to_timedelta(exp["day_offset"], unit="D")
        exp["week_start"] = exp["stay_date"] - pd.to_timedelta(exp["stay_date"].dt.weekday, unit="D")
        exp["arrival_date_week_number"] = exp["arrival_date_week_number"].astype(int)
        return exp

    def _mark_seasons_date(self, ax: Any, dates: pd.Series) -> None:
        md = pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values("date")
        md["iso"] = md["date"].dt.isocalendar().week.astype(int)
        md["is_summer"] = md["iso"].between(27, 35)
        m, d = md["date"].dt.month, md["date"].dt.day
        md["is_winter"] = ((m == 12) & (d >= 24)) | ((m == 1) & (d <= 6))
        for fc, cl, lb in [("is_summer", "#1f7a1f", "Summer holiday"), ("is_winter", "#b22222", "Winter holiday")]:
            sd = md[["date", fc]].copy()
            sd["gid"] = (sd[fc] != sd[fc].shift(1, fill_value=False)).cumsum()
            first = True
            for _, p in sd[sd[fc]].groupby("gid"):
                s = p["date"].iloc[0]; e = p["date"].iloc[-1] + pd.Timedelta(days=1)
                ax.axvspan(s, e, color=cl, alpha=0.14, label=lb if first else None)
                first = False

    def _mark_seasons_week(self, ax: Any, week_starts: pd.Series) -> None:
        md = pd.DataFrame({"week_start": pd.to_datetime(week_starts)}).sort_values("week_start")
        md["iso"] = md["week_start"].dt.isocalendar().week.astype(int)
        md["is_summer"] = md["iso"].between(27, 35)
        starts, ends = pd.to_datetime(md["week_start"]), pd.to_datetime(md["week_start"]) + pd.Timedelta(days=6)
        win = []
        for s, e in zip(starts, ends):
            y = int(s.year)
            win.append((s <= pd.Timestamp(year=y, month=1, day=6)) and (e >= pd.Timestamp(year=y - 1, month=12, day=24)) or
                       (s <= pd.Timestamp(year=y + 1, month=1, day=6)) and (e >= pd.Timestamp(year=y, month=12, day=24)))
        md["is_winter"] = win
        for fc, cl, lb in [("is_summer", "#1f7a1f", "Summer holiday"), ("is_winter", "#b22222", "Winter holiday")]:
            sd = md[["week_start", fc]].copy()
            sd["gid"] = (sd[fc] != sd[fc].shift(1, fill_value=False)).cumsum()
            first = True
            for _, p in sd[sd[fc]].groupby("gid"):
                s = p["week_start"].iloc[0]; e = p["week_start"].iloc[-1] + pd.Timedelta(days=7)
                ax.axvspan(s, e, color=cl, alpha=0.14, label=lb if first else None)
                first = False

    def _plot_children_weekly(self, ax: Any, hotel: str, hotel_df: pd.DataFrame) -> None:
        fw = hotel_df["week_start"].min(); lw = hotel_df["week_start"].max()
        aw = pd.date_range(fw, lw, freq="7D")
        self._mark_seasons_week(ax, aw)
        for gn, cl in [("Without children", "#1f7a1f"), ("With children", "#b22222")]:
            gs = hotel_df[hotel_df["guestGroup"] == gn].groupby("week_start").size().reindex(aw, fill_value=0)
            ax.plot(gs.index, gs.values, color=cl, linewidth=1.2, label=gn)
        ax.set_title(f"{hotel} - weekly occupancy")
        ax.set_xlim(left=pd.Timestamp('2015-07-01'), right=pd.Timestamp('2017-07-31'))
        ax.set_xlabel("Week start"); ax.set_ylabel("Occupied rooms"); ax.legend()

    def _plot_children_arrival_week(self, ax: Any, hotel: str, hotel_df: pd.DataFrame) -> None:
        wr = np.arange(1, 54)
        for gn, cl in [("Without children", "#1f7a1f"), ("With children", "#b22222")]:
            gs = hotel_df[hotel_df["guestGroup"] == gn].groupby("arrival_date_week_number").size().reindex(wr, fill_value=0)
            ax.plot(wr, gs.values, color=cl, linewidth=1.2, label=gn)
        ax.axvspan(27, 35, color="#1f7a1f", alpha=0.14, label="Summer holiday")
        ax.axvspan(51.5, 53.5, color="#b22222", alpha=0.14, label="Winter holiday")
        ax.axvspan(0.5, 1.5, color="#b22222", alpha=0.14)
        ax.set_title(f"{hotel} - by arrival_date_week_number")
        ax.set_xlabel("arrival_date_week_number"); ax.set_ylabel("Occupied rooms")
        ax.set_xlim(1, 53); ax.set_xticks(np.arange(1, 54, 4)); ax.legend()

    # ── Kids ADR (cell 50) ──────────────────────────────────────────

    def plot_kids_adr(self, data: pd.DataFrame) -> None:
        df = data.copy()
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
        df["children"] = df["children"].fillna(0); df["babies"] = df["babies"].fillna(0)
        df = df.dropna(subset=["hotel", "arrival_date", "adr"]).copy()
        df["kidsGroup"] = np.where((df["children"] + df["babies"]) > 0, "With kids", "Without kids")
        hotels = sorted(df["hotel"].dropna().unique())
        if len(hotels) != 2:
            raise ValueError(f"Expected 2 hotels, got {len(hotels)}")
        fig, axes = plt.subplots(2, 2, figsize=(18, 8))
        fig.suptitle("Mean and median ADR by reservations with kids vs without kids", fontsize=14, fontweight="bold")
        for ci, hotel in enumerate(hotels):
            hd = df[df["hotel"] == hotel].copy()
            self._plot_kids_weekly(axes[0, ci], hotel, hd)
            self._plot_kids_week_number(axes[1, ci], hotel, hd)
        fig.tight_layout(); plt.show()
        comp = self._kids_median_comparison(df)
        print("\nMedian ADR comparison by hotel: with kids vs without kids")
        print(comp.to_string(index=False))

    def _plot_kids_weekly(self, ax: Any, hotel: str, hotel_df: pd.DataFrame) -> None:
        w = hotel_df.groupby(["kidsGroup", pd.Grouper(key="arrival_date", freq="W-MON")])["adr"].agg(adr_mean="mean", adr_median="median").reset_index().rename(columns={"arrival_date": "week_start"}).sort_values("week_start")
        aw = pd.date_range(w["week_start"].min(), w["week_start"].max(), freq="7D")
        self._mark_seasons_week(ax, aw)
        for gn, cl in [("Without kids", "#1f7a1f"), ("With kids", "#b22222")]:
            gd = w[w["kidsGroup"] == gn]
            ax.plot(gd["week_start"], gd["adr_mean"], color=cl, linewidth=1.5, label=f"{gn} - mean ADR")
            ax.plot(gd["week_start"], gd["adr_median"], color=cl, linewidth=1.5, linestyle="--", label=f"{gn} - median ADR")
        ax.set_title(f"{hotel} - weekly interval")
        ax.set_xlim(left=pd.Timestamp('2015-07-01'), right=pd.Timestamp('2017-07-31'))
        ax.set_xlabel("Week start"); ax.set_ylabel("ADR"); ax.legend(fontsize=8)

    def _plot_kids_week_number(self, ax: Any, hotel: str, hotel_df: pd.DataFrame) -> None:
        w = hotel_df[["arrival_date_week_number", "kidsGroup", "adr"]].dropna().groupby(["arrival_date_week_number", "kidsGroup"])["adr"].agg(adr_mean="mean", adr_median="median").reset_index().sort_values("arrival_date_week_number")
        for gn, cl in [("Without kids", "#1f7a1f"), ("With kids", "#b22222")]:
            gd = w[w["kidsGroup"] == gn]
            ax.plot(gd["arrival_date_week_number"], gd["adr_mean"], color=cl, linewidth=1.5, label=f"{gn} - mean ADR")
            ax.plot(gd["arrival_date_week_number"], gd["adr_median"], color=cl, linewidth=1.5, linestyle="--", label=f"{gn} - median ADR")
        ax.axvspan(27, 35, color="#1f7a1f", alpha=0.14, label="Summer holiday")
        ax.axvspan(51.5, 53.5, color="#b22222", alpha=0.14, label="Winter holiday")
        ax.axvspan(0.5, 1.5, color="#b22222", alpha=0.14)
        ax.set_title(f"{hotel} - by week number")
        ax.set_xlabel("arrival_date_week_number"); ax.set_ylabel("ADR")
        ax.set_xlim(1, 53); ax.set_xticks(np.arange(1, 54, 4)); ax.legend(fontsize=8)

    def _kids_median_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        med = df.groupby(["hotel", "kidsGroup"])["adr"].median().unstack("kidsGroup")
        med["median_adr_without_kids"] = med.get("Without kids", float("nan"))
        med["median_adr_with_kids"] = med.get("With kids", float("nan"))
        comp = med[["median_adr_without_kids", "median_adr_with_kids"]].reset_index()
        comp["absolute_difference"] = comp["median_adr_with_kids"] - comp["median_adr_without_kids"]
        comp["relative_difference"] = np.where(comp["median_adr_without_kids"] != 0,
                                                comp["absolute_difference"] / comp["median_adr_without_kids"], float("nan"))
        comp["relative_difference_percent"] = comp["relative_difference"] * 100
        return comp[["hotel", "median_adr_without_kids", "median_adr_with_kids",
                      "absolute_difference", "relative_difference", "relative_difference_percent"]].round({
            "median_adr_without_kids": 2, "median_adr_with_kids": 2, "absolute_difference": 2,
            "relative_difference": 4, "relative_difference_percent": 2})

    # ── Room match ADR (cell 53) ───────────────────────────────────

    def plot_room_match_adr(self, data: pd.DataFrame) -> None:
        df = data.copy()
        df = df.loc[df["country"].notna() & df["children"].notna()].copy()
        df = df[df["adults"] != 0].copy()
        if "arrival_date" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["arrival_date"]):
            df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
        df = df.dropna(subset=["hotel", "arrival_date", "adr", "reserved_room_type", "assigned_room_type"]).copy()
        df["roomMatch"] = np.where(df["reserved_room_type"] == df["assigned_room_type"], "Same room type", "Different room type")
        print("Room match counts:\n", df["roomMatch"].value_counts().to_string())
        hotels = sorted(df["hotel"].unique())
        if len(hotels) != 2:
            raise ValueError(f"Expected 2 hotels, got {len(hotels)}")
        fig, axes = plt.subplots(2, 2, figsize=(18, 8))
        fig.suptitle("Mean and median ADR by room-type match (same vs different)", fontsize=14, fontweight="bold")
        for ci, hotel in enumerate(hotels):
            hd = df[df["hotel"] == hotel].copy()
            self._plot_room_match_weekly(axes[0, ci], hotel, hd)
            self._plot_room_match_week_number(axes[1, ci], hotel, hd)
        fig.tight_layout(); plt.show()

    def _plot_adr_lines(self, ax: Any, plot_data: pd.DataFrame, x_col: str) -> None:
        for mv, cl in [("Same room type", "#1f7a1f"), ("Different room type", "#b22222")]:
            md = plot_data[plot_data["roomMatch"] == mv]
            if md.empty:
                continue
            ax.plot(md[x_col], md["adr_mean"], color=cl, linewidth=1.5, label=f"{mv} - mean ADR")
            ax.plot(md[x_col], md["adr_median"], color=cl, linewidth=1.5, linestyle="--", label=f"{mv} - median ADR")

    def _plot_room_match_weekly(self, ax: Any, hotel: str, hotel_df: pd.DataFrame) -> None:
        w = hotel_df.groupby(["roomMatch", pd.Grouper(key="arrival_date", freq="W-MON")])["adr"].agg(adr_mean="mean", adr_median="median").reset_index().rename(columns={"arrival_date": "week_start"}).sort_values("week_start")
        aw = pd.date_range(w["week_start"].min(), w["week_start"].max(), freq="7D")
        self._mark_seasons_week(ax, aw)
        self._plot_adr_lines(ax, w, "week_start")
        ax.set_title(f"{hotel} - weekly interval")
        ax.set_xlim(left=pd.Timestamp('2015-07-01'), right=pd.Timestamp('2017-07-31'))
        ax.set_xlabel("Week start"); ax.set_ylabel("ADR"); ax.legend(fontsize=8)

    def _plot_room_match_week_number(self, ax: Any, hotel: str, hotel_df: pd.DataFrame) -> None:
        w = hotel_df[["arrival_date_week_number", "roomMatch", "adr"]].dropna().groupby(["arrival_date_week_number", "roomMatch"])["adr"].agg(adr_mean="mean", adr_median="median").reset_index().sort_values("arrival_date_week_number")
        self._plot_adr_lines(ax, w, "arrival_date_week_number")
        ax.axvspan(27, 35, color="#1f7a1f", alpha=0.14, label="Summer holiday")
        ax.axvspan(51.5, 53.5, color="#b22222", alpha=0.14, label="Winter holiday")
        ax.axvspan(0.5, 1.5, color="#b22222", alpha=0.14)
        ax.set_title(f"{hotel} - by week number")
        ax.set_xlabel("arrival_date_week_number"); ax.set_ylabel("ADR")
        ax.set_xlim(1, 53); ax.set_xticks(np.arange(1, 54, 4)); ax.legend(fontsize=8)

    # ── Lead-time scatter (cell 56, printADRPlots) ─────────────────

    def plot_lead_time_adr(self, data: pd.DataFrame, adr_quantiles: tuple[float, float] = (0.01, 0.99)) -> None:
        lo = float(data["adr"].quantile(adr_quantiles[0]))
        hi = float(data["adr"].quantile(adr_quantiles[1]))
        df = data[(data["adr"] >= lo) & (data["adr"] <= hi)].copy()
        fig, ax = plt.subplots(figsize=(24, 8))
        cancelled = df[df["reservation_status"].isin(["Canceled", "No-Show"])]
        not_cancelled = df[df["reservation_status"] == "Check-Out"]
        ax.scatter(cancelled["lead_time"], cancelled["adr"], label="Cancelled", c="#b22222", alpha=0.4, s=10)
        ax.scatter(not_cancelled["lead_time"], not_cancelled["adr"], label="Not Cancelled", c="#2e8b57", alpha=0.4, s=10)
        qq = pd.qcut(df["lead_time"], q=4, duplicates="drop", retbins=True)
        bins_ = qq[1]
        for i, b in enumerate(bins_):
            if i == 0 or i == len(bins_) - 1:
                continue
            ax.axvline(b, color="#f68115", linestyle="-", linewidth=5, alpha=1)
        labels_q = ["Q1", "Q2", "Q3", "Q4"]
        for i in range(len(bins_) - 1):
            mp = (bins_[i] + bins_[i + 1]) / 2
            if labels_q[i] != "Q4":
                ax.text(mp, ax.get_ylim()[1] * 0.95, labels_q[i], ha="center", va="top", fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            else:
                ax.text(mp - 150, ax.get_ylim()[1] * 0.95, labels_q[i], ha="center", va="top", fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        lead_bins = np.arange(0, df["lead_time"].max() + 10, 10)
        centers = (lead_bins[:-1] + lead_bins[1:]) / 2
        means, medians = [], []
        for i in range(len(lead_bins) - 1):
            bd = df[(df["lead_time"] >= lead_bins[i]) & (df["lead_time"] < lead_bins[i + 1])]["adr"]
            means.append(bd.mean() if len(bd) > 0 else float("nan"))
            medians.append(bd.median() if len(bd) > 0 else float("nan"))
        ax.plot(centers, means, color="purple", linewidth=5, label="Mean ADR", alpha=0.7)
        ax.plot(centers, medians, color="blue", linewidth=5, label="Median ADR", alpha=0.7)
        ax.set_xlabel("lead_time"); ax.set_ylabel("adr")
        ax.set_title("ADR by lead time")
        ax.set_xlim(left=-5, right=600); ax.legend(); ax.grid(True, linewidth=0.5); ax.set_axisbelow(True)
        fig.tight_layout(); plt.show()

    # ── Market segment occupancy (cell 59) ─────────────────────────

    def plot_market_segment_occupancy(self, data: pd.DataFrame) -> None:
        occ = self._prepare_market_segment_base(data)
        exp = self._expand_market_segment_daily(occ)
        hotels = sorted(exp["hotel"].dropna().unique())
        if len(hotels) != 2:
            raise ValueError(f"Expected 2 hotels, got {len(hotels)}")
        fig, axes = plt.subplots(2, 2, figsize=(18, 8))
        fig.suptitle("Occupancy by market segment (weekly and arrival week number)", fontsize=14, fontweight="bold")
        for ci, hotel in enumerate(hotels):
            hd = exp[exp["hotel"] == hotel].copy()
            self._plot_market_weekly(axes[0, ci], hotel, hd)
            self._plot_market_arrival_week(axes[1, ci], hotel, hd)
        fig.tight_layout(); plt.show()

    def _prepare_market_segment_base(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
        df["total_nights"] = (df["stays_in_week_nights"].fillna(0) + df["stays_in_weekend_nights"].fillna(0)).astype(int)
        df["marketSegment"] = df["market_segment"].fillna("Unknown")
        occ = df[["hotel", "marketSegment", "arrival_date", "arrival_date_week_number", "total_nights"]].copy()
        occ = occ[~occ["marketSegment"].isin(["Complementary", "Aviation", "Contemporary"])].copy()
        occ = occ.dropna(subset=["hotel", "arrival_date", "arrival_date_week_number"])
        bt = occ.groupby(["hotel", "marketSegment"]).size().rename("count").reset_index()
        bt["percentage"] = (bt["count"] / bt.groupby("hotel")["count"].transform("sum") * 100).round(2)
        rn = occ.groupby(["hotel", "marketSegment"])["total_nights"].sum().rename("room_nights").reset_index()
        bt = bt.merge(rn, on=["hotel", "marketSegment"], how="left")
        bt["room_nights_share_percent"] = (bt["room_nights"] / bt.groupby("hotel")["room_nights"].transform("sum") * 100).round(2)
        bt = bt.sort_values(["hotel", "room_nights", "marketSegment"], ascending=[True, False, True]).reset_index(drop=True)
        print(bt.to_string(index=False))
        return occ[occ["total_nights"] > 0].copy()

    def _expand_market_segment_daily(self, occ: pd.DataFrame) -> pd.DataFrame:
        exp = occ.loc[occ.index.repeat(occ["total_nights"])].copy()
        exp["day_offset"] = exp.groupby(level=0).cumcount()
        exp["stay_date"] = exp["arrival_date"] + pd.to_timedelta(exp["day_offset"], unit="D")
        exp["week_start"] = exp["stay_date"] - pd.to_timedelta(exp["stay_date"].dt.weekday, unit="D")
        exp["arrival_date_week_number"] = exp["arrival_date_week_number"].astype(int)
        return exp

    def _plot_market_weekly(self, ax: Any, hotel: str, hotel_df: pd.DataFrame) -> None:
        fw = hotel_df["week_start"].min(); lw = hotel_df["week_start"].max()
        aw = pd.date_range(fw, lw, freq="7D")
        self._mark_seasons_week(ax, aw)
        segs = sorted(hotel_df["marketSegment"].dropna().unique())
        pal = sns.color_palette("tab10", n_colors=max(len(segs), 1))
        for i, seg in enumerate(segs):
            gs = hotel_df[hotel_df["marketSegment"] == seg].groupby("week_start").size().reindex(aw, fill_value=0)
            ax.plot(gs.index, gs.values, color=pal[i % len(pal)], linewidth=1.2, label=seg)
        ax.set_title(f"{hotel} - weekly occupancy by market_segment")
        ax.set_xlabel("Week start")
        ax.set_xlim(left=pd.Timestamp('2015-07-01'), right=pd.Timestamp('2017-07-31'))
        ax.set_ylabel("Occupied rooms"); ax.legend(fontsize=8)

    def _plot_market_arrival_week(self, ax: Any, hotel: str, hotel_df: pd.DataFrame) -> None:
        wr = np.arange(1, 54)
        segs = sorted(hotel_df["marketSegment"].dropna().unique())
        pal = sns.color_palette("tab10", n_colors=max(len(segs), 1))
        for i, seg in enumerate(segs):
            gs = hotel_df[hotel_df["marketSegment"] == seg].groupby("arrival_date_week_number").size().reindex(wr, fill_value=0)
            ax.plot(wr, gs.values, color=pal[i % len(pal)], linewidth=1.2, label=seg)
        ax.axvspan(27, 35, color="#1f7a1f", alpha=0.14, label="Summer holiday")
        ax.axvspan(51.5, 53.5, color="#b22222", alpha=0.14, label="Winter holiday")
        ax.axvspan(0.5, 1.5, color="#b22222", alpha=0.14)
        ax.set_title(f"{hotel} - by arrival_date_week_number")
        ax.set_xlabel("arrival_date_week_number"); ax.set_ylabel("Occupied rooms")
        ax.set_xlim(1, 53); ax.set_xticks(np.arange(1, 54, 4)); ax.legend(fontsize=8)

    # ── Market segment ADR (cell 60) ────────────────────────────────

    def plot_market_segment_adr(self, data: pd.DataFrame) -> None:
        df = data.copy()
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
        df = df.dropna(subset=["hotel", "arrival_date", "adr", "market_segment"]).copy()
        df["marketSegment"] = df["market_segment"].fillna("Unknown")
        df = df[~df["marketSegment"].isin(["Complementary", "Aviation", "Contemporary"])].copy()
        hotels = sorted(df["hotel"].dropna().unique())
        if len(hotels) != 2:
            raise ValueError(f"Expected 2 hotels, got {len(hotels)}")
        fig, axes = plt.subplots(4, 2, figsize=(20, 18))
        fig.suptitle("Mean and median ADR by market segment", fontsize=16, fontweight="bold")
        for ci, hotel in enumerate(hotels):
            hd = df[df["hotel"] == hotel].copy()
            self._plot_market_adr_week(axes[0, ci], hotel, hd, "mean")
            self._plot_market_adr_week(axes[1, ci], hotel, hd, "median")
            self._plot_market_adr_week_number(axes[2, ci], hotel, hd, "mean")
            self._plot_market_adr_week_number(axes[3, ci], hotel, hd, "median")
        axes[0, 0].set_ylabel("ADR"); axes[1, 0].set_ylabel("ADR")
        axes[2, 0].set_ylabel("ADR"); axes[3, 0].set_ylabel("ADR")
        fig.tight_layout(); plt.show()
        comp = self._market_median_comparison(df)
        print("\nMarket segment median ADR comparison by hotel (vs hotel median ADR):")
        print(comp.to_string(index=False))

    def _plot_market_adr_week(self, ax: Any, hotel: str, hotel_df: pd.DataFrame, vt: str) -> None:
        w = hotel_df.groupby(["marketSegment", pd.Grouper(key="arrival_date", freq="W-MON")])["adr"].agg(adr_mean="mean", adr_median="median").reset_index().rename(columns={"arrival_date": "week_start"}).sort_values("week_start")
        aw = pd.date_range(w["week_start"].min(), w["week_start"].max(), freq="7D")
        self._mark_seasons_week(ax, aw)
        segs = sorted(w["marketSegment"].dropna().unique())
        pal = sns.color_palette("tab10", n_colors=max(len(segs), 1))
        ls = "--" if vt == "median" else "-"
        yc = "adr_median" if vt == "median" else "adr_mean"
        for i, seg in enumerate(segs):
            sd = w[w["marketSegment"] == seg]
            ax.plot(sd["week_start"], sd[yc], color=pal[i % len(pal)], linewidth=1.5, linestyle=ls, label=f"{seg} - {vt} ADR")
        ax.set_title(f"{hotel} - weekly {vt}")
        ax.set_xlabel("Week start"); ax.set_ylabel("ADR"); ax.legend(fontsize=8)

    def _plot_market_adr_week_number(self, ax: Any, hotel: str, hotel_df: pd.DataFrame, vt: str) -> None:
        w = hotel_df[["arrival_date_week_number", "marketSegment", "adr"]].dropna().groupby(["arrival_date_week_number", "marketSegment"])["adr"].agg(adr_mean="mean", adr_median="median").reset_index().sort_values("arrival_date_week_number")
        segs = sorted(w["marketSegment"].dropna().unique())
        pal = sns.color_palette("tab10", n_colors=max(len(segs), 1))
        ls = "--" if vt == "median" else "-"
        yc = "adr_median" if vt == "median" else "adr_mean"
        for i, seg in enumerate(segs):
            sd = w[w["marketSegment"] == seg]
            ax.plot(sd["arrival_date_week_number"], sd[yc], color=pal[i % len(pal)], linewidth=1.5, linestyle=ls, label=f"{seg} - {vt} ADR")
        ax.axvspan(27, 35, color="#1f7a1f", alpha=0.14, label="Summer holiday")
        ax.axvspan(51.5, 53.5, color="#b22222", alpha=0.14, label="Winter holiday")
        ax.axvspan(0.5, 1.5, color="#b22222", alpha=0.14)
        ax.set_title(f"{hotel} - {vt} by arrival_date_week_number")
        ax.set_xlabel("arrival_date_week_number"); ax.set_ylabel("ADR")
        ax.set_xlim(1, 53); ax.set_xticks(np.arange(1, 54, 4)); ax.legend(fontsize=8)

    def _market_median_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        hotel_med = df.groupby("hotel")["adr"].median().rename("hotel_median_adr").reset_index()
        seg_med = df.groupby(["hotel", "marketSegment"])["adr"].median().rename("segment_median_adr").reset_index()
        comp = seg_med.merge(hotel_med, on="hotel", how="left")
        comp["absolute_difference"] = comp["segment_median_adr"] - comp["hotel_median_adr"]
        comp["relative_difference"] = np.where(comp["hotel_median_adr"] != 0,
                                                comp["absolute_difference"] / comp["hotel_median_adr"], float("nan"))
        comp["relative_difference_percent"] = comp["relative_difference"] * 100
        comp = comp.sort_values(["hotel", "segment_median_adr"], ascending=[True, False]).reset_index(drop=True)
        return comp[["hotel", "marketSegment", "segment_median_adr", "hotel_median_adr",
                      "absolute_difference", "relative_difference", "relative_difference_percent"]].round({
            "segment_median_adr": 2, "hotel_median_adr": 2, "absolute_difference": 2,
            "relative_difference": 4, "relative_difference_percent": 2})
