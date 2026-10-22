from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import os
import joblib


app = Flask(__name__)
CORS(app)


# ==================================================
# Paths
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PERFORMANCE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "performance",
    "performance_model.pkl"
)


# ==================================================
# Load Performance Model
# ==================================================

performance_model = None

try:

    if os.path.exists(
        PERFORMANCE_MODEL_PATH
    ):

        performance_model = joblib.load(
            PERFORMANCE_MODEL_PATH
        )

        print(
            "Student Performance Model loaded successfully."
        )

    else:

        print(
            "Performance model not found."
        )

except Exception as error:

    print(
        "Performance Model Loading Error:",
        error
    )


# ==================================================
# Course Recommendation
# ==================================================

def get_course_text(course):

    title = course.get(
        "title",
        ""
    )

    description = course.get(
        "description",
        ""
    )

    category = course.get(
        "category",
        ""
    )

    level = course.get(
        "level",
        ""
    )

    return (
        f"{title} "
        f"{title} "
        f"{description} "
        f"{category} "
        f"{category} "
        f"{level}"
    )


# ==================================================
# Recommendation API
# ==================================================

@app.post("/recommend")
def recommend_courses():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "message": "Request data is required."
            }), 400


        courses = data.get(
            "courses",
            []
        )


        enrolled_course_ids = data.get(
            "enrolledCourseIds",
            []
        )


        if not courses:

            return jsonify({
                "success": True,
                "message": "No courses available.",
                "data": []
            })


        enrolled_course_ids = {
            str(course_id)
            for course_id in enrolled_course_ids
        }


        available_courses = [

            course

            for course in courses

            if str(
                course.get("_id")
                or course.get("id")
                or course.get("courseId")
            )

            not in enrolled_course_ids

        ]


        if not available_courses:

            return jsonify({
                "success": True,
                "message": "No new courses available.",
                "data": []
            })


        course_documents = [

            get_course_text(course)

            for course in courses

        ]


        available_documents = [

            get_course_text(course)

            for course in available_courses

        ]


        vectorizer = TfidfVectorizer(

            stop_words="english",

            ngram_range=(1, 2)

        )


        all_vectors = vectorizer.fit_transform(
            course_documents
        )


        available_vectors = vectorizer.transform(
            available_documents
        )


        enrolled_courses = [

            course

            for course in courses

            if str(
                course.get("_id")
                or course.get("id")
                or course.get("courseId")
            )

            in enrolled_course_ids

        ]


        if not enrolled_courses:

            return jsonify({
                "success": True,
                "message": "No learning history available.",
                "data": []
            })


        enrolled_documents = [

            get_course_text(course)

            for course in enrolled_courses

        ]


        enrolled_vectors = vectorizer.transform(
            enrolled_documents
        )


        student_profile = np.asarray(

            enrolled_vectors.mean(
                axis=0
            )

        )


        similarity_scores = cosine_similarity(

            student_profile,

            available_vectors

        )[0]


        recommendations = []


        for index, course in enumerate(
            available_courses
        ):

            course_id = str(

                course.get("_id")

                or course.get("id")

                or course.get("courseId")

            )


            similarity = float(

                similarity_scores[index]

            )


            recommendations.append({

                "courseId":
                    course_id,

                "title":
                    course.get(
                        "title",
                        ""
                    ),

                "description":
                    course.get(
                        "description",
                        ""
                    ),

                "image":
                    course.get(
                        "image",
                        ""
                    ),

                "category":
                    course.get(
                        "category",
                        ""
                    ),

                "level":
                    course.get(
                        "level",
                        ""
                    ),

                "duration":
                    course.get(
                        "duration",
                        ""
                    ),

                "faculty":
                    course.get(
                        "faculty",
                        ""
                    ),

                "courseType":
                    course.get(
                        "courseType",
                        ""
                    ),

                "price":
                    course.get(
                        "price",
                        0
                    ),

                "students":
                    course.get(
                        "students",
                        0
                    ),

                "similarity":
                    round(
                        similarity,
                        4
                    ),

                "score":
                    round(
                        similarity * 100,
                        2
                    )

            })


        recommendations.sort(

            key=lambda item:
                item["similarity"],

            reverse=True

        )


        recommendations = [

            course

            for course in recommendations

            if course["similarity"] >= 0.05

        ]


        recommendations = recommendations[:6]


        return jsonify({

            "success":
                True,

            "message":
                "ML recommendations generated successfully.",

            "model":
                "Weighted TF-IDF + Cosine Similarity",

            "data":
                recommendations

        })


    except Exception as error:

        print(
            "Recommendation ML Error:",
            error
        )


        return jsonify({

            "success":
                False,

            "message":
                "Unable to generate ML recommendations.",

            "error":
                str(error)

        }), 500


# ==================================================
# Student Performance Prediction API
# ==================================================

@app.post("/predict-performance")
def predict_performance():

    try:

        # --------------------------------------------------
        # Check model
        # --------------------------------------------------

        if performance_model is None:

            return jsonify({

                "success": False,

                "message":
                    "Performance prediction model is not available."

            }), 500


        # --------------------------------------------------
        # Get request data
        # --------------------------------------------------

        data = request.get_json()


        if not data:

            return jsonify({

                "success": False,

                "message":
                    "Request data is required."

            }), 400


        # --------------------------------------------------
        # Required fields
        # --------------------------------------------------

        required_fields = [

            "courseProgress",

            "completedLessons",

            "totalLessons",

            "quizAverage",

            "quizAttempts",

            "passedQuizzes"

        ]


        missing_fields = [

            field

            for field in required_fields

            if field not in data

        ]


        if missing_fields:

            return jsonify({

                "success": False,

                "message":
                    "Required fields are missing.",

                "missingFields":
                    missing_fields

            }), 400


        # --------------------------------------------------
        # Convert values to numbers
        # --------------------------------------------------

        course_progress = float(
            data["courseProgress"]
        )

        completed_lessons = int(
            data["completedLessons"]
        )

        total_lessons = int(
            data["totalLessons"]
        )

        quiz_average = float(
            data["quizAverage"]
        )

        quiz_attempts = int(
            data["quizAttempts"]
        )

        passed_quizzes = int(
            data["passedQuizzes"]
        )


        # --------------------------------------------------
        # Basic validation
        # --------------------------------------------------

        if not 0 <= course_progress <= 100:

            return jsonify({

                "success": False,

                "message":
                    "courseProgress must be between 0 and 100."

            }), 400


        if total_lessons < 0:

            return jsonify({

                "success": False,

                "message":
                    "totalLessons cannot be negative."

            }), 400


        if completed_lessons < 0:

            return jsonify({

                "success": False,

                "message":
                    "completedLessons cannot be negative."

            }), 400


        if completed_lessons > total_lessons:

            return jsonify({

                "success": False,

                "message":
                    "completedLessons cannot be greater than totalLessons."

            }), 400


        if not 0 <= quiz_average <= 100:

            return jsonify({

                "success": False,

                "message":
                    "quizAverage must be between 0 and 100."

            }), 400


        if quiz_attempts < 0:

            return jsonify({

                "success": False,

                "message":
                    "quizAttempts cannot be negative."

            }), 400


        if passed_quizzes < 0:

            return jsonify({

                "success": False,

                "message":
                    "passedQuizzes cannot be negative."

            }), 400


        if passed_quizzes > quiz_attempts:

            return jsonify({

                "success": False,

                "message":
                    "passedQuizzes cannot be greater than quizAttempts."

            }), 400


        # --------------------------------------------------
        # Create model input
        # --------------------------------------------------

        input_data = [[

            course_progress,

            completed_lessons,

            total_lessons,

            quiz_average,

            quiz_attempts,

            passed_quizzes

        ]]


        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        prediction = performance_model.predict(
            input_data
        )[0]


        # --------------------------------------------------
        # Prediction probabilities
        # --------------------------------------------------

        probabilities = {}


        if hasattr(
            performance_model,
            "predict_proba"
        ):

            probability_values = (

                performance_model.predict_proba(
                    input_data
                )[0]

            )


            for class_name, probability in zip(

                performance_model.classes_,

                probability_values

            ):

                probabilities[
                    str(class_name)
                ] = round(

                    float(probability) * 100,

                    2

                )


        # --------------------------------------------------
        # Response
        # --------------------------------------------------

        return jsonify({

            "success":
                True,

            "prediction":
                str(prediction),

            "probabilities":
                probabilities,

            "model":
                "Random Forest Classifier",

            "features": {

                "courseProgress":
                    course_progress,

                "completedLessons":
                    completed_lessons,

                "totalLessons":
                    total_lessons,

                "quizAverage":
                    quiz_average,

                "quizAttempts":
                    quiz_attempts,

                "passedQuizzes":
                    passed_quizzes

            }

        })


    except ValueError:

        return jsonify({

            "success": False,

            "message":
                "All performance values must be valid numbers."

        }), 400


    except Exception as error:

        print(
            "Performance Prediction Error:",
            error
        )


        return jsonify({

            "success": False,

            "message":
                "Unable to predict student performance.",

            "error":
                str(error)

        }), 500


# ==================================================
# Health Check
# ==================================================

@app.get("/")
def home():

    return jsonify({

        "success":
            True,

        "message":
            "LearnDesk ML Service is running.",

        "models": [

            "Weighted TF-IDF + Cosine Similarity",

            "Random Forest Classifier"

        ]

    })


# ==================================================
# Start Flask Server
# ==================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=8000,

        debug=True

    )