from django.shortcuts import render
from django.views import View
from .models import UserUpload
import os

import tensorflow as tf
import os
import cv2
import numpy as np
from keras.models import load_model

# Create your views here.

#    
#     
#     Ustrasana (Camel Pose)
#     Benefits: Stretches chest, abdomen, and hip flexors; strengthens back and shoulders; may improve posture.
#     Tips: Keep your hips over your knees, engage your core, and avoid crunching your lower back.

#     Utthita Trikonasana (Extended Triangle Pose)
#     Benefits: Stretches hamstrings and hips; strengthens legs and core; improves balance.
#     Tips: Keep your front knee slightly bent, engage your core, and maintain a long spine.

#     Viparita Karani (Legs-Up-the-Wall Pose)
#     Benefits: Relieves stress; reduces swelling in the legs; gently stretches hamstrings and lower back.
#     Tips: Use a folded blanket under your hips for support, relax your upper body, and breathe deeply.

#     Adho Mukha Svanasana (Downward-Facing Dog)
#     Benefits: Stretches hamstrings, calves, and shoulders; strengthens arms and legs; energizes the body.
#     Tips: Keep your spine long, bend your knees if necessary, and press firmly into your hands and feet.

#     Balasana (Child's Pose)
#     Benefits: Stretches hips, thighs, and lower back; calms the mind; relieves stress.
#     Tips: Widen your knees for a deeper stretch, use a prop under your forehead if needed, and breathe deeply.

#     Bhujangasana (Cobra Pose)
#     Benefits: Strengthens spine; stretches chest, shoulders, and abdomen; may relieve lower back pain.
#     Tips: Keep your elbows close to your body, engage your back muscles, and avoid straining your neck.

#     Phalakasana (Plank Pose)
#     Benefits: Strengthens arms, shoulders, core, and legs; improves posture and balance.
#     Tips: Keep your body in a straight line, engage your core, and press firmly into your hands.

#     Tadasana (Mountain Pose)
#     Benefits: Promotes balance and alignment; strengthens legs and core; improves posture.
#     Tips: Stand with feet hip-width apart, engage your core, and keep your spine long and shoulders relaxed.

#     Trikonasana (Triangle Pose)
#     Benefits: Stretches hamstrings and hips; strengthens legs and core; improves balance and stability.
#     Tips: Keep your front knee slightly bent, engage your core, and maintain a long spine while reaching for the ground or a block.

#     Virabhadrasana I (Warrior I)
#     Benefits: Strengthens legs and glutes; stretches hip flexors and chest; improves balance.
#     Tips: Keep your front knee over your ankle, square your hips to the front, and engage your core.

#     Virabhadrasana II (Warrior II)
#     Benefits: Strengthens legs and glutes; stretches hips and groin; improves balance and stability.
#     Tips: Keep your front knee over your ankle, engage your core, and maintain a strong, upright torso.

#     Vrksasana (Tree Pose)
#     Benefits: Strengthens legs and ankles; improves balance, focus, and concentration.
#     Tips: Press your foot into your inner thigh or calf, engage your core, and focus on a steady point in front of you.
# }


class HomeView(View):
    def get(self, request, *args, **kwargs):
        return render(request, "home/index.html")
    
    def post(self, request, *args, **kwargs):
        user_upload = request.FILES.get("user_upload")
        user_class = request.POST.get("user_class")
        new_upload = UserUpload.objects.create(image=user_upload)

        print(new_upload.image.url)

        # Define the categories
        yoga_categories = ["Chaturanga_Dandasana", "Crescent_Lunge", "Janu_Sirsasana", "Malasana", "Reclining_Supported_Twist", "Salamba_Sirsasana_II", "Ustrasana", "Utthita_Trikonasana", "Viparita_Karani", "Adho_Mukha_Svanasana", "Balasana", "Bhujangasana", "Phalakasana", "Tadasana", "Trikonasana", "Virabhadrasana_I", "Virabhadrasana_II", "Vrksasana"]

        # Define the categories
        gym_categories = ["abdominal_crunche","benchpress", "bicep_curls", "cable_curls", "chest_flys", "deadlift", "dipsgym","lat_pulldown","legs_press","planks_gym_post","pullups","russian_twist","shoulderpress","squats", "tricep_extensions"]

        # Define the categories
        daily_categories = ["bending", "brushing_teeth", "climbidaily_categoriesng_stairs", "cooking", "cycling", "kneeling","lifting","lying_down","pulling","pushing","reading","running","sitting","squating","standing","sweeping","typing","walking","washing hands"]

        #Img Size
        img_size = 100

        #Prediction function
        def yoga_predict(image_path, model):
            img = cv2.imread(image_path)
            img = cv2.resize(img, (img_size, img_size))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)
            prediction_category = yoga_categories[np.argmax(prediction)]
            prediction_accuracy = np.max(prediction)

            print(prediction_accuracy)

            if prediction_accuracy > 0.6:
                return prediction_category
            else:
                return "Cannot process the image properly, try another posture"
        
        #Prediction function
        def gym_predict(image_path, model):
            img = cv2.imread(image_path)
            img = cv2.resize(img, (img_size, img_size))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)
            prediction_category = gym_categories[np.argmax(prediction)]
            prediction_accuracy = np.max(prediction)

            print(prediction_accuracy)

            if prediction_accuracy > 0.6:
                return prediction_category
            else:
                return "Cannot process the image properly, try another posture"
        
        #Prediction function
        def daily_predict(image_path, model):
            img = cv2.imread(image_path)
            img = cv2.resize(img, (img_size, img_size))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)
            prediction_category = daily_categories[np.argmax(prediction)]
            prediction_accuracy = np.max(prediction)

            print(prediction_accuracy)

            if prediction_accuracy > 0.6:
                return prediction_category
            else:
                return "Cannot process the image properly, try another posture"
        
        print(user_class)

        if user_class == "yoga":
            #load model
            new_model = load_model('home/koksi_yoga_01.h5')

            #Make prediction
            img_path = os.path.join("media", f"{user_upload}")
            print(img_path)
            category = yoga_predict(img_path, new_model)

        elif user_class == "daily":
            #load model
            new_model = load_model('home/koksi_daily_activities_01.h5')

            #Make prediction
            img_path = os.path.join("media", f"{user_upload}")
            print(img_path)
            category = daily_predict(img_path, new_model)

        else:
            #load model
            new_model = load_model('home/koksi_gym_01.h5')

            #Make prediction
            img_path = os.path.join("media", f"{user_upload}")
            print(img_path)
            category = gym_predict(img_path, new_model)
        

        context = {
            "res": category,
            "new_upload":new_upload
        }
        return render(request, "home/result.html", context)