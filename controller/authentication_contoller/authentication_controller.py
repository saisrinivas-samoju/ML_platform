import os
import sys
from flask import  render_template, redirect, url_for, jsonify, session
from flask import request

from entity_layer.registration.registration import Register

global process_value

class AuthenticationController:
    def __init__(self):
        pass

    def validate_email_address(self):
        """
        Checks if the given email address is valid or not
        """
        try:
            email_address = request.form['email_address']  # Takes the email address given
            if email_address is not None:
                reg = Register()
                response = reg.is_email_address_allowed(email_address=email_address)
                if not response['status']:
                    return jsonify({'status': True, 'message': response['message'], })
                response = reg.is_email_address_used(email_address=email_address)
                return jsonify(response)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            return jsonify({'status': False, 'message': str(exception_detail)})

    def register(self):
        """
        For registering new user
        """
        try:
            if 'email_address' in session:
                # If the user has already logged in, redirect to the index page
                return redirect(url_for('index'))
            # If the user is not logged in
            if request.method == "POST":
                # Capture the details entered by the user
                user_name = request.form['user_name']
                password = request.form['password']
                confirm_password = request.form['confirm_password']
                email_address = request.form['email_address']
                # And register user
                reg = Register()
                response = reg.register_user(user_name, email_address, password, confirm_password)
                # If registration is successful, go to login page
                if response['status']:
                    return render_template("login.html",
                                           context={'message': response['message'], 'message_status': 'success'})
                # Else reload the register page
                else:
                    return render_template('register.html',
                                           context={'message': response['message'], 'message_status': 'danger'})
            else:
                return render_template("register.html", context={'message': None, 'message_status': 'info'})

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            return render_template('error.html',
                                   context={'message': None, 'message_status': 'info', 'error_message': str(exception_detail)})



    def login(self):
        try:
            # If email_address is not in session, take it from the user input.
            if 'email_address' not in session:
                if request.method == "POST":
                    reg = Register()
                    email_address = request.form['email_address']
                    password = request.form['password']
                    response = reg.verify_user(email_address=email_address, password=password)
                    # If the user is verified, take the email address, and go to index page
                    if response['status']:
                        session['email_address'] = email_address
                        return render_template("index.html",
                                               context={'message': response['message'], 'message_status': 'success'})
                    # Llse, revert back to login page to try again
                    else:
                        return render_template("login.html",
                                               context={'message': response['message'], 'message_status': 'danger'})
                else:
                    return render_template("login.html", context={'message': None, 'message_status': 'info'})
            return redirect(url_for('index'))
            #return render_template("login.html", context={'message': None, 'message_status': 'info'})

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            return render_template('error.html',
                                   context={'message': None, 'message_status': 'info', 'error_message': str(exception_detail)})

    def logout(self):
        try:
            # If the email_address is in session i.e. logged in, remove the email address from the session i.e. logout, and redirect to index page
            if 'email_address' in session:
                session.pop('email_address', None)
            return redirect(url_for('index'))

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            return render_template('error.html',
                                   context={'message': None, 'message_status': 'info', 'error_message': str(exception_detail)})
