from flask import Flask, render_template, request, redirect, url_for, session, flash
from config import Config
from webapp.services.api_client import APIClient

app = Flask(__name__)
app.config.from_object(Config)


# ROUTES UI

@app.route("/")
def home():
    if "access_token" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        response = APIClient.login(username, password)

        if response.status_code == 200:
            data = response.json()
            session["access_token"] = data["access_token"]
            flash("Connexion réussie !", "success")
            return redirect(url_for("home"))
        else:
            flash("Erreur de connexion", "danger")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        response = APIClient.register(username, email, password)

        if response.status_code == 200:
            flash("Compte créé !", "success")
            return redirect(url_for("login"))
        else:
            flash("Erreur lors de l'inscription", "danger")

    return render_template("register.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "access_token" not in session:
        return redirect(url_for("login"))

    file = request.files["audio_file"]
    response = APIClient.predict(file)

    if response.status_code == 401:
        session.clear()
        flash("Session expirée", "warning")
        return redirect(url_for("login"))

    if response.status_code == 200:
        result = response.json()
        return render_template("result.html", result=result)

    flash("Erreur lors de la prédiction", "danger")
    return redirect(url_for("home"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)