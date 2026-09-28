from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("trading", "0004_add_nifty_smallcap250_flag"),
    ]

    operations = [
        migrations.AddField(
            model_name="stock",
            name="is_nifty500",
            field=models.BooleanField(db_index=True, default=False),
        ),
    ]
