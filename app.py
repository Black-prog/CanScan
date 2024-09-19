import re
from flask import Flask, render_template, redirect, url_for, request, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy import desc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
from datetime import datetime
from fpdf import FPDF
from werkzeug.utils import secure_filename
from flask import send_file
from flask import make_response
import os

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo
model = load_model('modelo_deteccion_cancer_inceptionv3.h5')

# Configuración de la base de datos
app.config['SECRET_KEY'] = 'davidm123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Modelos de la base de datos
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)
    profile_pic = db.Column(db.String(200), nullable=True, default="Default_user.png")  # Nueva columna

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    patient_name = db.Column(db.String(150), nullable=False)
    date = db.Column(db.String(50), nullable=False)
    condition = db.Column(db.String(150), nullable=False)
    image_path = db.Column(db.String(150), nullable=False)

# Crear tablas
with app.app_context():
    db.create_all()

# Configuración de la carpeta de subida
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Rutas de la aplicación
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Validar email con una expresión regular
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash('El correo electrónico no tiene un formato válido', 'error')
            return redirect(url_for('register'))

        # Validar que las contraseñas coincidan
        if password != confirm_password:
            flash('Las contraseñas no coinciden', 'error')
            return redirect(url_for('register'))

        # Guardar el usuario en la base de datos si todo es válido
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        # Redirigir al perfil o a donde quieras
        session['user_id'] = new_user.id
        return redirect(url_for('profile'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Buscar el usuario en la base de datos por correo electrónico
        user = User.query.filter_by(email=email).first()

        # Verificar las credenciales
        if user and user.password == password:
            # Credenciales correctas, iniciar sesión
            session['user_id'] = user.id
            return redirect(url_for('profile'))
        else:
            # Credenciales incorrectas, mostrar mensaje de error
            flash('Credenciales incorrectas', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/profile')
def profile():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))

    user = User.query.filter_by(id=user_id).first()

    return render_template('profile.html', user=user)

@app.route('/update-profile-pic', methods=['POST'])
def update_profile_pic():
    user_id = session.get('user_id')
    user = User.query.get(user_id)

    if 'file' not in request.files:
        flash('No se ha seleccionado un archivo', 'error')
        return redirect(url_for('profile'))

    file = request.files['file']
    if file.filename == '':
        flash('No se ha seleccionado un archivo', 'error')
        return redirect(url_for('profile'))

    # Validar que el archivo subido es una imagen permitida
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Eliminar la imagen anterior si existe y no es la imagen predeterminada
        if user.profile_pic and user.profile_pic != 'uploads/Default_user.png':
            old_image_path = os.path.join(app.config['UPLOAD_FOLDER'], user.profile_pic)
            if os.path.exists(old_image_path):
                os.remove(old_image_path)

        # Guardar la nueva imagen en el servidor
        file.save(filepath)

        # Guardar la ruta completa de la nueva imagen en el campo 'profile_pic' del usuario
        user.profile_pic = 'uploads/' + filename  # Guardar la ruta relativa dentro de 'static/uploads'
        db.session.commit()

        flash('Foto de perfil actualizada con éxito', 'success')
        return redirect(url_for('profile'))
    else:
        flash('Formato de archivo no permitido. Solo se aceptan imágenes (png, jpg, jpeg, gif)', 'error')
        return redirect(url_for('profile'))

@app.route('/histories')
def histories():
    user_id = session.get('user_id')
    histories = History.query.filter_by(user_id=user_id).limit(4).all()
    return render_template('histories.html', histories=histories)

@app.route('/start-analysis', methods=['POST'])
def start_analysis():
    return redirect(url_for('index'))

@app.route('/index')
def index():
    # Obtener el user_id de la sesión
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))

    # Obtener el usuario logueado
    user = User.query.filter_by(id=user_id).first()

    return render_template('index.html', user=user)

@app.route('/all-histories')
def all_histories():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))

    search_query = request.args.get('search_query', '')  # Obtener el valor de la búsqueda

    if search_query:
        # Filtrar historias por el nombre del paciente según la búsqueda
        histories = History.query.filter_by(user_id=user_id).filter(History.patient_name.ilike(f'%{search_query}%')).all()
    else:
        # Mostrar todas las historias si no hay búsqueda
        histories = History.query.filter_by(user_id=user_id).all()

    return render_template('all_histories.html', histories=histories)

@app.route('/download-pdf/<int:history_id>', methods=['GET'])
def download_analysis_pdf(history_id):
    # Buscar la historia por ID
    history = History.query.get(history_id)
    
    if history:
        # Crear un PDF
        pdf = FPDF()
        pdf.add_page()

        # Título del análisis
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(200, 10, txt="Resultado del Análisis", ln=True, align='C')

        # Nombre del paciente
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(200, 10, txt=f"Paciente: {history.patient_name}", ln=True)

        # Fecha del análisis
        pdf.set_font('Arial', '', 12)
        pdf.cell(200, 10, txt=f"Fecha: {history.date}", ln=True)

        # Condición predicha
        pdf.cell(200, 10, txt=f"Condición Predicha: {history.condition}", ln=True)

        # Cargar la imagen
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(history.image_path))
        if os.path.exists(image_path):
            # Ajustar la imagen dentro del PDF
            pdf.image(image_path, x=10, y=80, w=100)  # Puedes ajustar 'x', 'y' y 'w' según sea necesario
        else:
            pdf.cell(200, 10, txt="Imagen no disponible", ln=True)

        # Guardar el PDF en un archivo temporal
        output_path = f"Resultado_Analisis_{history.patient_name}.pdf"
        pdf.output(output_path)

        # Descargar el PDF
        return send_file(output_path, as_attachment=True)

    else:
        return "No se encontró la historia médica.", 404

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("Has cerrado sesión correctamente", "success")
    return redirect(url_for('login'))

@app.route('/download-pdf')
def download_pdf():
    file_path = 'static/formato_entrada.pdf'  # Ruta al archivo PDF
    return send_file(file_path, as_attachment=True)

# Ruta para mostrar el formulario de edición de perfil
@app.route('/edit-profile', methods=['GET'])
def edit_profile():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    
    user = User.query.get(user_id)
    return render_template('edit_profile.html', user=user)

# Ruta para manejar la actualización del perfil
@app.route('/update-profile', methods=['POST'])
def update_profile():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    
    user = User.query.get(user_id)
    
    # Actualizar los datos del usuario
    user.username = request.form['username']
    user.email = request.form['email']
    
    # Solo actualizamos la contraseña si el campo no está vacío
    new_password = request.form['password']
    if new_password:
        user.password = new_password
    
    # Guardar los cambios en la base de datos
    db.session.commit()
    
    flash('Perfil actualizado exitosamente', 'success')
    return redirect(url_for('profile'))

@app.route('/delete-account', methods=['POST'])
def delete_account():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))

    # Obtener al usuario y sus historias médicas
    user = User.query.get(user_id)
    histories = History.query.filter_by(user_id=user_id).all()

    try:
        # Eliminar todas las historias médicas del usuario
        for history in histories:
            db.session.delete(history)

        # Eliminar al usuario
        db.session.delete(user)
        db.session.commit()

        # Cerrar sesión y redirigir al login
        session.pop('user_id', None)
        flash('Tu cuenta ha sido eliminada con éxito', 'success')
        return redirect(url_for('login'))

    except Exception as e:
        db.session.rollback()  # Revertir cambios en caso de error
        flash(f'Error al eliminar la cuenta: {str(e)}', 'error')
        return redirect(url_for('edit_profile'))

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Obtener los datos del formulario
        doctor_name = request.form.get('doctor_name')
        patient_first_name = request.form.get('patient_first_name')
        patient_last_name = request.form.get('patient_last_name')
        patient_full_name = f"{patient_first_name} {patient_last_name}"

        if 'file' not in request.files:
            flash('No se ha seleccionado un archivo', 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No se ha seleccionado un archivo', 'error')
            return redirect(request.url)

        if file:
            # Guardar la imagen subida
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            print(f"Imagen guardada en {filepath}")

            # Preprocesar la imagen
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            print(f"Forma de la imagen después del preprocesamiento: {img_array.shape}")

            # Hacer la predicción
            predictions = model.predict(img_array)
            print(f"Predicciones: {predictions}")
            classes = ['seborrheic_keratosis', 'nevus', 'melanoma']
            prediction_class = classes[np.argmax(predictions)]
            print(f"Clase predicha: {prediction_class}")

            # Obtener la fecha actual
            current_date = datetime.now().strftime('%d/%m/%Y %H:%M:%S')  # Formato: día/mes/año horas:minutos:segundos

            # Guardar los datos obtenidos en la base de datos
            user_id = session.get('user_id')
            condition = prediction_class  # Clase predicha
            image_path = filepath

            new_history = History(user_id=user_id, patient_name=patient_full_name, date=current_date, condition=condition, image_path=image_path)
            db.session.add(new_history)
            db.session.commit()

            return redirect(url_for('all_histories'))


    except Exception as e:
        # Registrar el error en la consola para depuración
        print(f"Error al analizar la imagen: {e}")
        flash(f"Error interno del servidor: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Guardar la imagen en la carpeta 'static/uploads/'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocesar la imagen
        img = image.load_img(filepath, target_size=(299, 299))  # Ajusta el tamaño según el modelo
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Hacer la predicción
        predictions = model.predict(img_array)
        classes = ['seborrheic_keratosis', 'nevus', 'melanoma']
        prediction_class = classes[np.argmax(predictions)]

        # Asegúrate de pasar la ruta correcta incluyendo 'uploads/' al template
        image_path = f'uploads/{file.filename}'

        return render_template('predict.html', prediction=prediction_class, image_path=image_path)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
