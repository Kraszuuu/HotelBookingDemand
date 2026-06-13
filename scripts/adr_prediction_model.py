from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ADRPredictionModel:
    def __init__(self):
        self.occupancy_context_: dict[str, Any] | None = None

    def remap_room_types(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        rules = {
            "City Hotel": {"Cheap": ["A", "B", "C", "K"], "Expensive": ["D", "E", "F", "G"]},
            "Resort Hotel": {"Cheap": ["A", "D", "I", "L"], "Expensive": ["B", "C", "E", "F", "G", "H"]},
        }
        for hotel, segments in rules.items():
            mask = df["hotel"] == hotel
            for seg_name, rooms in segments.items():
                df.loc[mask & df["assigned_room_type"].isin(rooms), "assigned_room_type"] = seg_name

        print("Updated assigned_room_type.")
        print(df.groupby(["hotel", "assigned_room_type"]).size().to_string())
        return df

    def _day_order(self) -> list[str]:
        return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def _prepare_occupancy_base(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
        df["total_nights"] = (df["stays_in_week_nights"].fillna(0) + df["stays_in_weekend_nights"].fillna(0)).astype(int)
        df = df[["hotel", "assigned_room_type", "arrival_date", "total_nights"]].copy()
        df = df.dropna(subset=["hotel", "arrival_date"])
        df["assigned_room_type"] = df["assigned_room_type"].fillna("Unknown")
        return df[df["total_nights"] > 0].copy()

    def _expand_to_daily(self, occ: pd.DataFrame) -> pd.DataFrame:
        expanded = occ.loc[occ.index.repeat(occ["total_nights"])].copy()
        expanded["day_offset"] = expanded.groupby(level=0).cumcount()
        expanded["stay_date"] = expanded["arrival_date"] + pd.to_timedelta(expanded["day_offset"], unit="D")
        expanded["week_start"] = expanded["stay_date"] - pd.to_timedelta(expanded["stay_date"].dt.weekday, unit="D")
        expanded["iso_week"] = expanded["stay_date"].dt.isocalendar().week.astype(int)
        expanded["day_name"] = pd.Categorical(
            expanded["stay_date"].dt.day_name(), categories=self._day_order(), ordered=True
        )
        return expanded

    def _build_date_ranges(self, expanded: pd.DataFrame) -> tuple[list[str], pd.DatetimeIndex, pd.DatetimeIndex]:
        hotels = sorted(expanded["hotel"].dropna().unique())
        min_d = expanded["stay_date"].min().normalize()
        max_d = expanded["stay_date"].max().normalize()
        all_dates = pd.date_range(min_d, max_d, freq="D")
        all_weeks = pd.date_range(min_d - pd.Timedelta(days=min_d.weekday()), max_d, freq="7D")
        return hotels, all_dates, all_weeks

    def _build_total_tables(
        self, expanded: pd.DataFrame, hotels: list[str], all_dates: pd.DatetimeIndex, all_weeks: pd.DatetimeIndex
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_daily = pd.MultiIndex.from_product([all_dates, hotels], names=["stay_date", "hotel"])
        full_weekly = pd.MultiIndex.from_product([all_weeks, hotels], names=["week_start", "hotel"])

        daily = (
            expanded.groupby(["stay_date", "hotel"]).size()
            .reindex(full_daily, fill_value=0).rename("occupied_rooms").reset_index()
        )
        daily["day_name"] = pd.Categorical(daily["stay_date"].dt.day_name(), categories=self._day_order(), ordered=True)
        daily["iso_week"] = daily["stay_date"].dt.isocalendar().week.astype(int)

        weekly = (
            expanded.groupby(["week_start", "hotel"]).size()
            .reindex(full_weekly, fill_value=0).rename("occupied_rooms").reset_index()
        )
        weekly["iso_week"] = weekly["week_start"].dt.isocalendar().week.astype(int)

        dow = daily.groupby(["day_name", "hotel"], observed=False)["occupied_rooms"].mean().reset_index()
        woy = weekly.groupby(["iso_week", "hotel"])["occupied_rooms"].mean().reset_index()
        return daily, weekly, dow, woy

    def _build_room_tables(self, expanded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        daily = (
            expanded.groupby(["stay_date", "hotel", "assigned_room_type"]).size()
            .rename("occupied_rooms").reset_index()
        )
        daily["day_name"] = pd.Categorical(daily["stay_date"].dt.day_name(), categories=self._day_order(), ordered=True)
        daily["iso_week"] = daily["stay_date"].dt.isocalendar().week.astype(int)

        weekly = (
            expanded.groupby(["week_start", "hotel", "assigned_room_type"]).size()
            .rename("occupied_rooms").reset_index()
        )
        weekly["iso_week"] = weekly["week_start"].dt.isocalendar().week.astype(int)

        dow = daily.groupby(["day_name", "hotel", "assigned_room_type"], observed=False)["occupied_rooms"].mean().reset_index()
        woy = weekly.groupby(["iso_week", "hotel", "assigned_room_type"])["occupied_rooms"].mean().reset_index()
        return daily, weekly, dow, woy

    def _build_calendar_flags(self, all_dates: pd.DatetimeIndex, all_weeks: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
        cal = pd.DataFrame({"stay_date": all_dates})
        cal["iso_week"] = cal["stay_date"].dt.isocalendar().week.astype(int)
        cal["is_summer"] = cal["iso_week"].between(27, 35)
        month, day = cal["stay_date"].dt.month, cal["stay_date"].dt.day
        cal["is_winter"] = ((month == 12) & (day >= 24)) | ((month == 1) & (day <= 6))
        cal["week_start"] = cal["stay_date"] - pd.to_timedelta(cal["stay_date"].dt.weekday, unit="D")
        weekly = cal.groupby("week_start")[["is_summer", "is_winter"]].max().reindex(all_weeks, fill_value=False)
        weekly.index.name = "week_start"
        weekly = weekly.reset_index()
        return cal, weekly

    def prepare_context(self, data: pd.DataFrame) -> dict[str, Any]:
        occ = self._prepare_occupancy_base(data)
        expanded = self._expand_to_daily(occ)
        hotels, all_dates, all_weeks = self._build_date_ranges(expanded)
        daily_t, weekly_t, dow_t, woy_t = self._build_total_tables(expanded, hotels, all_dates, all_weeks)
        daily_r, weekly_r, dow_r, woy_r = self._build_room_tables(expanded)
        cal, weekly_flags = self._build_calendar_flags(all_dates, all_weeks)
        ctx = {
            "hotelValues": hotels, "dailyTotal": daily_t, "weeklyTotal": weekly_t,
            "dayOfWeekTotal": dow_t, "weekOfYearTotal": woy_t,
            "dailyByRoom": daily_r, "weeklyByRoom": weekly_r,
            "dayOfWeekByRoom": dow_r, "weekOfYearByRoom": woy_r,
            "calendarDf": cal, "weeklyFlags": weekly_flags,
        }
        self.occupancy_context_ = ctx
        return ctx

    def _add_marked_ranges(self, ax: Any, values: pd.Series, color: str, alpha: float, label: str,
                           flags: pd.Series | None = None, is_numeric: bool = False) -> None:
        if flags is not None:
            df = pd.DataFrame({"value": pd.to_datetime(values), "flag": flags}).sort_values("value")
            df["start"] = df["flag"] & ~df["flag"].shift(1, fill_value=False)
            df["gid"] = df["start"].cumsum()
            first = True
            for _, part in df.loc[df["flag"]].groupby("gid"):
                s = part["value"].iloc[0]
                e = part["value"].iloc[-1] + pd.Timedelta(days=1)
                ax.axvspan(s, e, color=color, alpha=alpha, label=label if first else None)
                first = False
            return
        if is_numeric:
            vals = np.array(sorted({int(w) for w in values if pd.notna(w)}), dtype=int)
            if len(vals) == 0:
                return
            splits = np.where(np.diff(vals) > 1)[0] + 1
            first = True
            for part in np.split(vals, splits):
                ax.axvspan(int(part[0]) - 0.5, int(part[-1]) + 0.5, color=color, alpha=alpha,
                           label=label if first else None)
                first = False

    def _mark_seasons_by_date(self, ax: Any, cal: pd.DataFrame) -> None:
        self._add_marked_ranges(ax, cal["stay_date"], "#2e8b57", 0.12, "Summer holiday", flags=cal["is_summer"])
        self._add_marked_ranges(ax, cal["stay_date"], "#b22222", 0.12, "Winter holiday", flags=cal["is_winter"])

    def _mark_seasons_by_week(self, ax: Any, weekly_flags: pd.DataFrame) -> None:
        s = weekly_flags.loc[weekly_flags["is_summer"], "week_start"].dt.isocalendar().week.astype(int)
        w = weekly_flags.loc[weekly_flags["is_winter"], "week_start"].dt.isocalendar().week.astype(int)
        self._add_marked_ranges(ax, s, "#2e8b57", 0.12, "Summer holiday", is_numeric=True)
        self._add_marked_ranges(ax, w, "#b22222", 0.12, "Winter holiday", is_numeric=True)

    def plot_weekly_occupancy(self) -> None:
        ctx = self.occupancy_context_
        if ctx is None:
            raise RuntimeError("Call prepare_context() first")
        hotels = ctx["hotelValues"]
        weekly = ctx["weeklyByRoom"]
        flags = ctx["weeklyFlags"]
        fig, axes = plt.subplots(1, len(hotels), figsize=(8 * len(hotels), 4))
        axes = np.atleast_1d(axes)
        for idx, hotel in enumerate(hotels):
            ax = axes[idx]
            hf = weekly.loc[weekly["hotel"] == hotel].copy()
            self._mark_seasons_by_date(ax, flags.rename(columns={"week_start": "stay_date"}))
            for room in sorted(hf["assigned_room_type"].dropna().unique()):
                rd = hf.loc[hf["assigned_room_type"] == room].sort_values("week_start")
                color = "#1f7a1f" if room == "Cheap" else "#b22222" if room == "Expensive" else "#2e8b57"
                ax.plot(rd["week_start"], rd["occupied_rooms"], linewidth=1.2, color=color, label=room)
            ax.set_title(f"{hotel} - weekly occupancy by room type")
            ax.set_xlabel("Week start")
            ax.set_ylabel("Occupied rooms")
            ax.legend(title="Room type", ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot_iso_week_occupancy(self) -> None:
        ctx = self.occupancy_context_
        if ctx is None:
            raise RuntimeError("Call prepare_context() first")
        hotels = ctx["hotelValues"]
        woy = ctx["weekOfYearByRoom"]
        flags = ctx["weeklyFlags"]
        fig, axes = plt.subplots(1, len(hotels), figsize=(8 * len(hotels), 4))
        axes = np.atleast_1d(axes)
        for idx, hotel in enumerate(hotels):
            ax = axes[idx]
            hf = woy.loc[woy["hotel"] == hotel].copy()
            self._mark_seasons_by_week(ax, flags)
            for room in sorted(hf["assigned_room_type"].dropna().unique()):
                rd = hf.loc[hf["assigned_room_type"] == room]
                color = "#1f7a1f" if room == "Cheap" else "#b22222" if room == "Expensive" else "#2e8b57"
                ax.plot(rd["iso_week"], rd["occupied_rooms"], linewidth=1.5, color=color, label=room)
            ax.set_title(f"{hotel} - mean occupancy by ISO week")
            ax.set_xlabel("ISO week")
            ax.set_ylabel("Mean occupied rooms")
            ax.set_xlim(1, 53)
            ax.set_xticks(np.arange(1, 54, 4))
            ax.legend(title="Room type", ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot_weekday_occupancy(self) -> None:
        ctx = self.occupancy_context_
        if ctx is None:
            raise RuntimeError("Call prepare_context() first")
        hotels = ctx["hotelValues"]
        dow = ctx["dayOfWeekByRoom"]
        fig, axes = plt.subplots(1, len(hotels), figsize=(8 * len(hotels), 4))
        axes = np.atleast_1d(axes)
        for idx, hotel in enumerate(hotels):
            ax = axes[idx]
            hf = dow.loc[dow["hotel"] == hotel].copy()
            for room in sorted(hf["assigned_room_type"].dropna().unique()):
                rd = hf.loc[hf["assigned_room_type"] == room]
                color = "#1f7a1f" if room == "Cheap" else "#b22222" if room == "Expensive" else "#2e8b57"
                ax.plot(rd["day_name"].astype(str), rd["occupied_rooms"], marker="o", linewidth=1.5, color=color, label=room)
            ax.set_title(f"{hotel} - mean occupancy by weekday")
            ax.set_xlabel("Weekday")
            ax.set_ylabel("Mean occupied rooms")
            ax.legend(title="Room type", ncol=2, fontsize=8)
            ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.show()

    def plot_occupancy_by_room_type(self, data: pd.DataFrame) -> None:
        self.prepare_context(data)
        self.plot_weekly_occupancy()

    def plot_average_occupancy_by_room_type(self, data: pd.DataFrame) -> None:
        if self.occupancy_context_ is None:
            self.prepare_context(data)
        self.plot_iso_week_occupancy()
        self.plot_weekday_occupancy()
