from flask import Flask, render_template, request, url_for, flash, session, redirect
import os
import data_analysis

app = Flask(__name__)
ALLOWED_EXTENSIONS = ["csv"]
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath( __file__ )), "static")
USERNAME = "admin"
PASSWORD = "gt"
SECRET_KEY = "development key"

app.config.from_object(__name__)

def allowed_file(filename):
    return ('.' in filename) and (filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS)

def find_csv_filenames( suffix= ".csv" ):
    # path_to_dir = os.path.dirname(os.path.abspath(__file__))
    filenames = os.listdir(UPLOAD_FOLDER)
    return [ filename for filename in filenames if filename.endswith(suffix) ]

@app.route('/')
def upload_file():
    return render_template("home_screen.html", fig = url_for("static", filename = "heart_beat.gif"))

@app.route("/login", methods = ["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form["username"] != app.config["USERNAME"]:
            error = "Invalid Username"
        elif request.form["password"] != app.config["PASSWORD"]:
            error = "Invalid Password"
        else:
            session["logged_in"] = True
            session["file_name"] = None
            flash(message = "You are now logged in")
            return render_template("file_upload.html")
    return render_template("login.html", error = error, fig = url_for("static", filename = "skeleton.gif"))

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    session.pop("file_name", None)
    flash(message = "You're now logged out")
    return redirect(url_for("upload_file"))

@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        # check if the post request has the file part
        session["success"] = False
        if 'file' not in request.files:
            flash("No file part")
            return render_template("file_upload.html")
        # if user does not select file, browser also
        # submit a empty part without filename
        file = request.files["file"]
        if file.filename == '':
            flash('No selected file')
            return render_template("file_upload.html")
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # flash("File Uploaded")
            session["success"] = True
            return redirect(url_for("csvfiles"))
            # return render_template("show_entries.html")
        else:
            flash("Wrong filetype")
            return render_template("file_upload.html")

@app.route("/csvfiles")
def csvfiles():
    file_names = find_csv_filenames()
    entries = [dict(file = index, text = row) for index, row in enumerate(file_names)]
    flash("File Uploaded")
    return render_template("show_entries.html", entries = entries)

@app.route("/analyze", methods = ["POST"])
def analyze():
    # file_name = None
    if request.method == "POST":
        file_name = request.form["file_name"]
        session["file_name"] = file_name
        return render_template("orig.html", data_file = url_for("static", filename=file_name))
    else:
        return redirect(url_for("upload_file"))

# @app.route('/about')
# def about():
#     return render_template('about.html')

@app.route('/moving_average', methods = ["POST"])
def moving_average():
    data = data_analysis.read_data(session["file_name"])
    frequency = data_analysis.calc_freq_rate(data)
    # print "Frequency of the data is: ", frequency
    filtered = data_analysis.butter_lowpass_filter(data["hart"], 2.5, frequency, 5)
    roll_mean_data = data_analysis.rolling_mean(data, window_size=0.75, frequency=frequency)
    roll_mean_data.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "temp", "moving_avg.csv"),
                          index = False)
    return render_template("data_and_average.html", data_file = url_for("static", filename = "temp/moving_avg.csv"))

@app.route('/peaks')
def r_complex():
    data = data_analysis.read_data(session["file_name"])
    frequency = data_analysis.calc_freq_rate(data)
    # print "Frequency of the data is: ", frequency
    filtered = data_analysis.butter_lowpass_filter(data["hart"], 2.5, frequency, 5)
    roll_mean_data = data_analysis.rolling_mean(data, window_size=0.75, frequency=frequency)
    roll_mean_data.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "moving_avg.csv"),
                          index=False)
    data_analysis.fit_peaks(roll_mean_data, frequency)
    data_analysis.calc_frequency_measures(roll_mean_data, frequency)
    data_analysis.time_measures["bpm"] = (len(data_analysis.signal_measures["R_positions"]) / (len(roll_mean_data["hart"]) / frequency) * 60)
    R_positions = data_analysis.signal_measures["R_positions"]
    ybeat = data_analysis.signal_measures["R_values"]
    return render_template("peaks.html", data_file=url_for("static", filename="moving_avg.csv"))

if __name__ == '__main__':
    app.run(debug=True)