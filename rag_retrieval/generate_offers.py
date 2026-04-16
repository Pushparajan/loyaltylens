"""Generate 200 synthetic loyalty offers and write them to data/offers.json."""

from __future__ import annotations

import json
import random
from pathlib import Path

CATEGORIES = ["food", "beverage", "merchandise", "bonus_stars", "seasonal"]
CHANNELS = ["mobile", "all", "in-store"]

_TITLES: dict[str, list[str]] = {
    "food": [
        "Free Pastry with Any Purchase",
        "Double Points on Bakery Items",
        "Buy 2 Sandwiches Get 1 Free",
        "Loyalty Breakfast Bundle",
        "Weekend Brunch Bonus",
        "Snack Attack Deal",
        "Lunch Combo Saver",
        "Healthy Bites Reward",
    ],
    "beverage": [
        "Free Espresso Shot Upgrade",
        "Happy Hour Smoothie Discount",
        "Cold Brew Loyalty Perk",
        "Tea Time Triple Points",
        "Seasonal Latte Bonus",
        "Juice Bar Reward",
        "Sparkling Water Freebie",
        "Hot Chocolate Holiday Deal",
    ],
    "merchandise": [
        "Reusable Cup Discount",
        "Branded Tote Bag Offer",
        "Loyalty Tumbler Deal",
        "Gift Card Bonus Points",
        "Merchandise Bundle Reward",
        "Eco Bag Double Points",
        "Travel Mug Loyalty Perk",
        "Apparel Member Discount",
    ],
    "bonus_stars": [
        "5x Stars on Next Purchase",
        "Birthday Bonus Stars",
        "Milestone Reward Unlock",
        "Referral Stars Bonus",
        "Weekend Star Multiplier",
        "App-Exclusive Star Boost",
        "Monthly Star Challenge",
        "Tier Upgrade Star Boost",
    ],
    "seasonal": [
        "Summer Refreshment Deal",
        "Winter Warmer Bonus",
        "Spring Bloom Reward",
        "Autumn Harvest Perk",
        "Holiday Season Offer",
        "New Year Loyalty Bonus",
        "Valentine's Day Special",
        "Back-to-School Deal",
    ],
}

_DESCRIPTIONS: dict[str, list[str]] = {
    "food": [
        (
            "Enjoy a complimentary pastry of your choice when you spend $10 or more. "
            "This offer is valid on freshly baked items only and excludes packaged goods. "
            "Redeem in a single transaction at any participating location."
        ),
        (
            "Earn double loyalty points on every bakery purchase this month. "
            "Points are credited within 24 hours of your transaction. "
            "Combine with other food offers for maximum rewards."
        ),
        (
            "Purchase any two full-priced sandwiches and receive the third item free. "
            "Discount applies to the lowest-priced sandwich automatically. "
            "Valid for dine-in and take-away orders."
        ),
    ],
    "beverage": [
        (
            "Upgrade your espresso to a double shot at no extra charge on your next visit. "
            "Simply show this offer at the counter before ordering. "
            "One upgrade per customer per day."
        ),
        (
            "Get 20% off any smoothie between 2 pm and 5 pm on weekdays. "
            "Offer applies to regular and large sizes across the full smoothie menu. "
            "Not valid with other beverage promotions."
        ),
        (
            "Enjoy triple loyalty points when you order cold brew coffee this week. "
            "Points are awarded per unit purchased with no cap on daily transactions. "
            "Available at all café locations nationwide."
        ),
    ],
    "merchandise": [
        (
            "Purchase a branded reusable cup and save 15% on the retail price. "
            "Use your new cup on the same visit to earn an additional 50 bonus stars. "
            "Helping the environment has never been more rewarding."
        ),
        (
            "Members receive an exclusive discount on our limited-edition tote bag. "
            "Available while stocks last in-store and through the mobile app. "
            "Each bag purchased contributes to our sustainability fund."
        ),
        (
            "Redeem points towards a branded travel mug at a reduced star cost this season. "
            "Choose from three colours exclusive to loyalty programme members. "
            "Offer valid until end of the current quarter."
        ),
    ],
    "bonus_stars": [
        (
            "Earn 5x loyalty stars on your very next purchase after activating this offer. "
            "The multiplier applies to the full transaction value including food and beverages. "
            "Bonus stars are credited to your account within 48 hours."
        ),
        (
            "Celebrate your birthday month with a special star bonus on every purchase. "
            "Stars are automatically doubled throughout your birthday month at checkout. "
            "Birthday must be verified in your loyalty profile to qualify."
        ),
        (
            "Reach your next loyalty tier faster with a one-time star multiplier boost. "
            "Activated automatically when you are within 200 stars of your next tier. "
            "Contact support if the bonus does not appear within 72 hours."
        ),
    ],
    "seasonal": [
        (
            "Cool down this summer with 25% off any iced beverage of your choice. "
            "Offer runs from June through August at all participating stores. "
            "Stackable with your standard member discount for extra savings."
        ),
        (
            "Warm up this winter with a free syrup shot in any hot drink. "
            "Choose from vanilla, caramel, or hazelnut at no additional cost. "
            "Valid throughout December and January at all locations."
        ),
        (
            "Celebrate the new season with bonus stars on seasonal menu items. "
            "All limited-edition spring and autumn drinks qualify for double points. "
            "Seasonal menu items are highlighted in the app for easy identification."
        ),
    ],
}


def _pick(items: list[str], rng: random.Random) -> str:
    return rng.choice(items)


def generate_offers(n: int = 200, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    offers: list[dict] = []

    for i in range(1, n + 1):
        category = _pick(CATEGORIES, rng)
        title_pool = _TITLES[category]
        desc_pool = _DESCRIPTIONS[category]

        # Vary titles slightly to avoid exact duplicates across 200 offers
        base_title = _pick(title_pool, rng)
        suffix = rng.choice(["", " — Members Only", " — Limited Time", " — App Exclusive", ""])
        title = base_title + suffix

        description = _pick(desc_pool, rng)

        offers.append(
            {
                "id": f"offer_{i:04d}",
                "title": title,
                "description": description,
                "category": category,
                "channel": _pick(CHANNELS, rng),
                "min_propensity_threshold": round(rng.uniform(0.0, 0.7), 2),
                "discount_pct": rng.randint(5, 40),
                "expiry_days": rng.choice([7, 14, 30, 60, 90]),
            }
        )

    return offers


def main() -> None:
    out_path = Path(__file__).parent / "data" / "offers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    offers = generate_offers(200)
    out_path.write_text(json.dumps(offers, indent=2), encoding="utf-8")
    print(f"Wrote {len(offers)} offers to {out_path}")


if __name__ == "__main__":
    main()
