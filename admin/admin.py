from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from app import app, db
from app import User

admin = Admin(app, name='Admin Panel', template_mode='bootstrap3')
admin.add_view(ModelView(User, db.session))
