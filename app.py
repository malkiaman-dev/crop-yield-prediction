from flask import Flask, request, render_template, send_file, redirect, url_for
from io import BytesIO
from datetime import datetime
import pandas as pd
import pickle
import sklearn

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer
)

print("scikit-learn version:", sklearn.__version__)

# Load models
with open("dtr.pkl", "rb") as f:
    dtr = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Load dataset for suggestions
df = pd.read_csv("yield_df.csv")

available_areas = sorted(df["Area"].dropna().astype(str).str.strip().unique().tolist())
available_items = sorted(df["Item"].dropna().astype(str).str.strip().unique().tolist())

app = Flask(__name__)

# Store latest successful prediction for report download
latest_report_data = {}


def validate_inputs(year, rainfall, pesticides, avg_temp, area, item):
    errors = []

    if year < 1900 or year > 2100:
        errors.append("Year must be between 1900 and 2100.")

    if rainfall < 0:
        errors.append("Average rainfall cannot be negative.")

    if pesticides < 0:
        errors.append("Pesticides value cannot be negative.")

    if avg_temp < -50 or avg_temp > 60:
        errors.append("Average temperature must be between -50°C and 60°C.")

    if not area or len(area.strip()) < 2:
        errors.append("Please enter a valid country name.")

    if not item or len(item.strip()) < 2:
        errors.append("Please enter a valid crop name.")

    return errors


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        prediction_text=None,
        error_text=None,
        form_data=None,
        available_areas=available_areas,
        available_items=available_items
    )


@app.route("/predict", methods=["POST"])
def predict():
    global latest_report_data

    form_data = {
        "Year": request.form.get("Year", "").strip(),
        "average_rain_fall_mm_per_year": request.form.get("average_rain_fall_mm_per_year", "").strip(),
        "pesticides_tonnes": request.form.get("pesticides_tonnes", "").strip(),
        "avg_temp": request.form.get("avg_temp", "").strip(),
        "Area": request.form.get("Area", "").strip(),
        "Item": request.form.get("Item", "").strip()
    }

    try:
        if not all(form_data.values()):
            return render_template(
                "index.html",
                prediction_text=None,
                error_text="Please fill in all fields before predicting.",
                form_data=form_data,
                available_areas=available_areas,
                available_items=available_items
            )

        try:
            year = int(form_data["Year"])
            rainfall = float(form_data["average_rain_fall_mm_per_year"])
            pesticides = float(form_data["pesticides_tonnes"])
            avg_temp = float(form_data["avg_temp"])
        except ValueError:
            return render_template(
                "index.html",
                prediction_text=None,
                error_text="Year, rainfall, pesticides, and temperature must be valid numbers.",
                form_data=form_data,
                available_areas=available_areas,
                available_items=available_items
            )

        area = form_data["Area"].strip()
        item = form_data["Item"].strip()

        validation_errors = validate_inputs(year, rainfall, pesticides, avg_temp, area, item)
        if validation_errors:
            return render_template(
                "index.html",
                prediction_text=None,
                error_text=" ".join(validation_errors),
                form_data=form_data,
                available_areas=available_areas,
                available_items=available_items
            )

        features = pd.DataFrame([{
            "Year": year,
            "average_rain_fall_mm_per_year": rainfall,
            "pesticides_tonnes": pesticides,
            "avg_temp": avg_temp,
            "Area": area,
            "Item": item
        }])

        if hasattr(preprocessor, "feature_names_in_"):
            features = features[preprocessor.feature_names_in_]

        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features)
        result = round(float(prediction[0]), 2)

        prediction_text = f"Predicted yield is: {result} tonnes"

        latest_report_data = {
            "year": year,
            "rainfall": rainfall,
            "pesticides": pesticides,
            "temperature": avg_temp,
            "country": area,
            "crop_item": item,
            "predicted_yield": result,
            "report_date": datetime.now().strftime("%d %B %Y, %I:%M %p")
        }

        return render_template(
            "index.html",
            prediction_text=prediction_text,
            error_text=None,
            form_data={
                "Year": year,
                "average_rain_fall_mm_per_year": rainfall,
                "pesticides_tonnes": pesticides,
                "avg_temp": avg_temp,
                "Area": area,
                "Item": item
            },
            available_areas=available_areas,
            available_items=available_items
        )

    except ValueError as e:
        error_msg = str(e)

        if "unknown categories" in error_msg.lower():
            if "column 0" in error_msg:
                friendly_error = "Please choose a country from the suggestions."
            elif "column 1" in error_msg:
                friendly_error = "Please choose a crop from the suggestions."
            else:
                friendly_error = "Please check country and crop name."
        else:
            friendly_error = f"Input error: {error_msg}"

        return render_template(
            "index.html",
            prediction_text=None,
            error_text=friendly_error,
            form_data=form_data,
            available_areas=available_areas,
            available_items=available_items
        )

    except Exception:
        return render_template(
            "index.html",
            prediction_text=None,
            error_text="Something went wrong while generating the prediction. Please check your values and try again.",
            form_data=form_data,
            available_areas=available_areas,
            available_items=available_items
        )


@app.route("/clear", methods=["GET"])
def clear():
    global latest_report_data
    latest_report_data = {}
    return redirect(url_for("index"))


@app.route("/download-report", methods=["GET"])
def download_report():
    global latest_report_data

    if not latest_report_data:
        return "No report available. Please generate a prediction first.", 400

    crop_item = latest_report_data["crop_item"]
    country = latest_report_data["country"]
    year = latest_report_data["year"]
    rainfall = latest_report_data["rainfall"]
    pesticides = latest_report_data["pesticides"]
    temperature = latest_report_data["temperature"]
    predicted_yield = latest_report_data["predicted_yield"]
    report_date = latest_report_data["report_date"]

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#166534"),
        spaceAfter=14
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=16,
        alignment=TA_LEFT,
        spaceAfter=10
    )

    footer_style = ParagraphStyle(
        "FooterStyle",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#374151"),
        spaceAfter=6
    )

    elements = []

    title = Paragraph(
        f"{crop_item} Crop Yield Prediction Report for {country}",
        title_style
    )
    elements.append(title)
    elements.append(Spacer(1, 10))

    intro = Paragraph(
        f"""
        This report provides a structured prediction summary for <b>{crop_item}</b> cultivation in <b>{country}</b> for the year <b>{year}</b>.
        The detailed input values and prediction results are shown in the table below.
        """,
        body_style
    )
    elements.append(intro)

    summary = Paragraph(
        f"""
        Based on the provided environmental and agricultural conditions:<br/><br/>
        • Average Rainfall: <b>{rainfall}</b> mm/year<br/>
        • Pesticides Used: <b>{pesticides}</b> tonnes<br/>
        • Average Temperature: <b>{temperature}</b>°C<br/><br/>
        The predicted yield of <b>{crop_item}</b> in <b>{country}</b> is <b>{predicted_yield}</b> tonnes.
        """,
        body_style
    )
    elements.append(summary)
    elements.append(Spacer(1, 10))

    table_data = [
        ["Field", "Value"],
        ["Crop Item", str(crop_item)],
        ["Country", str(country)],
        ["Year", str(year)],
        ["Average Rainfall", f"{rainfall} mm/year"],
        ["Pesticides Used", f"{pesticides} tonnes"],
        ["Average Temperature", f"{temperature} °C"],
        ["Predicted Yield", f"{predicted_yield} tonnes"]
    ]

    report_table = Table(table_data, colWidths=[210, 290])

    report_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16a34a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#9ca3af")),
        ("BOX", (0, 0), (-1, -1), 1.2, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f0fdf4")]),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    elements.append(report_table)
    elements.append(Spacer(1, 16))

    explanation = Paragraph(
        """
        This prediction is generated using historical agricultural data and machine learning analysis,
        which helps estimate crop production under similar conditions.
        """,
        body_style
    )
    elements.append(explanation)

    closing = Paragraph(
        """
        This report can serve as a useful reference for future agricultural planning and decision-making.
        However, actual yield may vary due to unforeseen environmental, climatic, or other external factors.
        """,
        body_style
    )
    elements.append(closing)
    elements.append(Spacer(1, 12))

    footer_1 = Paragraph(f"<b>Generated on:</b> {report_date}", footer_style)
    footer_2 = Paragraph("<b>Prepared by:</b> Crop Yield Prediction System", footer_style)
    footer_3 = Paragraph("<b>Prediction Model:</b> Machine Learning-Based Analysis", footer_style)
    footer_4 = Paragraph("<b>Developed by:</b> Malki Aman", footer_style)

    elements.append(footer_1)
    elements.append(footer_2)
    elements.append(footer_3)
    elements.append(footer_4)

    doc.build(elements)
    buffer.seek(0)

    filename = f"{crop_item}_{country}_crop_yield_report.pdf".replace(" ", "_")

    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf"
    )


if __name__ == "__main__":
    app.run()