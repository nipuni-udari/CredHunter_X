DATABASE_PASSWORD = "S0meR4ndomP4ssw0rd!2024xyz"


def connect():
    return f"postgres://user:{DATABASE_PASSWORD}@localhost/db"
