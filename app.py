from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from threading import Lock
from markdown import markdown
import re

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')  # Fallback for development
app.jinja_env.filters['tojson'] = json.dumps  # Add tojson filter

# Configure DeepSeek API client
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
)

# Storage for streaming data with thread safety
stream_storage = {}
storage_lock = Lock()

# Generate the prompt for the DeepSeek API
def generate_fitness_prompt(form_data):
    return f"""Based on the following information:
- Gender: {form_data['gender']}
- Age: {form_data['age']}
- Height: {form_data['height_m']} m
- Current Weight: {form_data['weight_kg']} kg
- Goal Weight: {form_data['goal_weight_kg']} kg
- Goal: {form_data['goal']}
- Activity Level: {form_data['activity_level']}
- Exercise environment: {form_data['exercise_environment']}

Provide a fitness plan in two parts:

### PART 1: Detailed Reasoning for the Fitness Plan ###
In this section, thoroughly reason through the creation of the fitness plan by:
1. **Analyzing User Data**: Calculate the user's Body Mass Index (BMI) using the formula (weight_kg / (height_m * height_m)) and assess whether they are underweight, normal weight, overweight, or obese. Consider how their current weight compares to their goal weight.
2. **Evaluating Activity Level**: Interpret the user's activity level (e.g., sedentary, lightly active, moderately active, very active) and estimate their Total Daily Energy Expenditure (TDEE) conceptually (no need for precise numbers, but explain how activity influences energy needs).
3. **Tailoring to the Goal**: Explain how the fitness plan aligns with the user's specific goal (e.g., weight loss, muscle gain, general fitness). For weight loss, emphasize calorie deficit through exercise (e.g., added cardio); for muscle gain, focus on progressive overload and targeting muscle groups; for general fitness, balance strength and endurance.
4. **Considering Gender and Age**: Discuss how gender and age might influence the plan (e.g., muscle-building potential, metabolism, or injury risk).
5. **Plan Structure Rationale**: Justify the choice of exercises, body parts targeted, sets, reps, rest days, and cardio (if applicable). Explain why the plan includes 2 rest days per 7-day cycle and how week 1 serves as a template with slight variations for subsequent weeks.
6. **Achieving the Goal**: Outline how the 30-day plan will move the user toward their goal weight or fitness objective, considering their starting point and timeline.

Provide a clear, logical, and detailed explanation (at least 2000 words) that connects the user's data to the plan's design.

### PART 2: The Complete 30-Day Fitness Plan ###
Provide the plan in this exact format:

### FITNESS PLAN ###

DAY_1:
  BODY_PARTS: [Body Part 1, Body Part 2]
  EXERCISES:
    - NAME: [Exercise Name 1]
      SETS: [Sets]
      REPS: [Reps]
    - NAME: [Exercise Name 2]
      SETS: [Sets]
      REPS: [Reps]
    - NAME: [Exercise Name 3]
      SETS: [Sets]
      REPS: [Reps]
    - NAME: [Exercise Name 4]
      SETS: [Sets]
      REPS: [Reps]
    - NAME: [Exercise Name 5]
      SETS: [Sets]
      REPS: [Reps]

DAY_2:
  BODY_PARTS: [Body Part 1, Body Part 2]
  EXERCISES:
    - NAME: [Exercise Name 1]
      SETS: [Sets]
      REPS: [Reps]
    - NAME: [Exercise Name 2]
      SETS: [Sets]
      REPS: [Reps]
    - NAME: [Exercise Name 3]
      SETS: [Sets]
      REPS: [Reps]
    - NAME: [Exercise Name 4]
      SETS: [Sets]
      REPS: [Reps]
    - NAME: [Exercise Name 5]
      SETS: [Sets]
      REPS: [Reps]

Continue this exact format for ALL 30 DAYS (DAY_1 through DAY_30). 
**IMPORTANT**: You MUST include all 30 days in your response.
- Include 2 rest days for every 7 days (e.g., Days 6 and 7 as rest in week 1).for 30 days.
- Use week 1 as the template for the following weeks, but introduce slight variations (e.g., change exercise order, increase reps/sets, or swap one exercise for a similar one).
- If the user's goal is weight loss, add cardio (e.g., 20-30 minutes of running, cycling, or jumping jacks) to every workout day.
- Include exactly 5 exercises for each workout day. Each day should target 2 different body parts."""
# Calculate BMI for display
def calculate_bmi(height_m, weight_kg):
    return weight_kg / (height_m ** 2) if height_m > 0 else 0

# Main route to render the form
@app.route('/')
def index():
    return render_template('index.html')

# Handle form submission
@app.route('/submit-form', methods=['POST'])
def submit_form():
    form_data = request.get_json()

    # Convert height from feet/inches to meters
    height_feet = float(form_data['height_feet'])
    height_inches = float(form_data['height_inches'])
    total_inches = height_feet * 12 + height_inches
    height_m = total_inches * 0.0254

    # Convert weights from pounds to kilograms
    weight_lbs = float(form_data['weight_lbs'])
    weight_kg = weight_lbs * 0.453592
    goal_weight_lbs = float(form_data['goal_weight_lbs'])
    goal_weight_kg = goal_weight_lbs * 0.453592

    # Store converted data in session
    session['form_data'] = {
        'gender': form_data['gender'],
        'age': int(form_data['age']),
        'height_m': height_m,
        'weight_kg': weight_kg,
        'goal_weight_kg': goal_weight_kg,
        'goal': form_data['goal'],
        'activity_level': form_data['activity_level'],
        'exercise_environment': form_data['exercise_environment']
    }

    return jsonify(success=True, stream_url=url_for('stream_updates'))

# Review page to display reasoning and BMI
@app.route('/review')
def review():
    form_data = session.get('form_data')
    if not form_data:
        return redirect(url_for('index'))
    bmi = calculate_bmi(form_data['height_m'], form_data['weight_kg'])
    return render_template('review.html', 
                         bmi=bmi,
                         gender=form_data['gender'],
                         age=form_data['age'],
                         height_m=form_data['height_m'],
                         weight_kg=form_data['weight_kg'],
                         goal_weight_kg=form_data['goal_weight_kg'],
                         goal=form_data['goal'],
                         activity_level=form_data['activity_level'],
                         exercise_environment=form_data.get('exercise_environment', 'gym'))

# Stream updates from DeepSeek API
@app.route('/stream-updates')
def stream_updates():
    form_data = session.get('form_data')
    if not form_data:
        return "No form data found", 400

    stream_id = str(id(request))

    def generate():
        full_response = ""
        reasoning = ""
        delimiter_found = False
        last_sent_length = 0  # Track how much text we've already sent

        try:
            # Log the prompt for debugging
            prompt = generate_fitness_prompt(form_data)
            print(f"DEBUG - Sending prompt to DeepSeek API: {prompt}")
            
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=8000,  # Ensure we get a complete response
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                full_response += content
                if not delimiter_found:
                    if '### FITNESS PLAN ###' in full_response:
                        delimiter_found = True
                        parts = full_response.split('### FITNESS PLAN ###', 1)
                        reasoning = parts[0].strip()
                        # Log when delimiter is found
                        print(f"DEBUG - Delimiter found. Reasoning length: {len(reasoning)}")
                    else:
                        reasoning = full_response.strip()
                
                # Log every 500 characters for debugging
                if len(full_response) % 500 == 0:
                    print(f"DEBUG - Received {len(full_response)} characters so far")
                
                # Only send updates when we have new content to avoid empty messages
                if len(reasoning) > last_sent_length:
                    # Send the full reasoning text each time, not just the new part
                    # This ensures the client always has the complete text
                    print(f"DEBUG - Sending update with reasoning length: {len(reasoning)}")
                    yield f"data: {json.dumps({'reasoning': reasoning})}\n\n"
                    last_sent_length = len(reasoning)

            # Split response into reasoning and plan
            parts = full_response.split('### FITNESS PLAN ###', 1)
            reasoning = parts[0].strip()
            plan = parts[1].strip() if len(parts) > 1 else ""
            
            # Log final lengths for debugging
            print(f"DEBUG - Final full response length: {len(full_response)}")
            print(f"DEBUG - Final reasoning length: {len(reasoning)}")
            print(f"DEBUG - Final plan length: {len(plan)}")
            print(f"DEBUG - Last 50 chars of reasoning: {reasoning[-50:] if len(reasoning) > 50 else reasoning}")
            print(f"DEBUG - First 100 chars of plan: {plan[:100] if plan else 'No plan'}")
            print(f"DEBUG - Last 100 chars of plan: {plan[-100:] if len(plan) > 100 else plan}")
            
            # Check how many days are in the plan
            if plan:
                day_matches = re.finditer(r'DAY_(\d+):', plan)
                days = [int(match.group(1)) for match in day_matches]
                if days:
                    print(f"DEBUG - Plan contains days: {sorted(days)}")
                    print(f"DEBUG - Total days found: {len(days)}")
                    if max(days) < 28:
                        print(f"DEBUG - WARNING: Plan is incomplete. Expected 28 days, highest day found: {max(days)}")
                else:
                    print("DEBUG - No days found in plan")

            # Make sure we send the complete reasoning one last time
            if len(reasoning) > last_sent_length:
                print(f"DEBUG - Sending final reasoning update with length: {len(reasoning)}")
                yield f"data: {json.dumps({'reasoning': reasoning})}\n\n"

            # Store data thread-safely
            with storage_lock:
                stream_storage[stream_id] = {'reasoning': reasoning, 'plan': plan}

            yield f"data: {json.dumps({'complete': True, 'redirect': f'/save-and-redirect/{stream_id}'})}\n\n"
        except Exception as e:
            print(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Content-Type': 'text/event-stream',
        'Connection': 'keep-alive'
    })

# Save streamed data and redirect to plan
@app.route('/save-and-redirect/<stream_id>')
def save_and_redirect(stream_id):
    with storage_lock:
        data = stream_storage.pop(stream_id, None)
    if data:
        # Add detailed logging
        reasoning_length = len(data['reasoning']) if data['reasoning'] else 0
        plan_length = len(data['plan']) if data['plan'] else 0
        print(f"DEBUG - save_and_redirect: Found data for stream_id {stream_id}")
        print(f"DEBUG - save_and_redirect: Reasoning length: {reasoning_length}")
        print(f"DEBUG - save_and_redirect: Plan length: {plan_length}")
        
        if plan_length > 0:
            print(f"DEBUG - Plan first 100 chars: {data['plan'][:100]}")
            print(f"DEBUG - Plan last 100 chars: {data['plan'][-100:]}")
            
            # Check how many days are in the plan
            day_matches = re.finditer(r'DAY_(\d+):', data['plan'])
            days = [int(match.group(1)) for match in day_matches]
            if days:
                print(f"DEBUG - Plan contains days: {sorted(days)}")
                print(f"DEBUG - Total days found: {len(days)}")
            else:
                print("DEBUG - No days found in plan")
        
        session['fitness_reasoning'] = data['reasoning']
        session['fitness_plan'] = data['plan']
        return redirect(url_for('plan'))
    
    print(f"DEBUG - save_and_redirect: No data found for stream_id {stream_id}")
    return redirect(url_for('review'))

# Plan page to display the workout plan
@app.route('/plan')
def plan():
    fitness_plan = session.get('fitness_plan', '')
    form_data = session.get('form_data', {})
    
    if not fitness_plan:
        print("DEBUG - plan: No fitness plan found in session")
        return redirect(url_for('index'))
    
    print(f"DEBUG - plan: Fitness plan length: {len(fitness_plan)}")
    print(f"DEBUG - plan: First 100 chars: {fitness_plan[:100]}")
    print(f"DEBUG - plan: Last 100 chars: {fitness_plan[-100:]}")
    
    # Parse the workout plan into a structured format
    workout_plan = {}
    
    # Simple parser for the specific format we requested
    # Look for day patterns: DAY_X:
    day_matches = re.finditer(r'DAY_(\d+):', fitness_plan)
    day_match_list = list(day_matches)
    print(f"DEBUG - plan: Found {len(day_match_list)} day matches")
    
    # Reset the iterator
    day_matches = re.finditer(r'DAY_(\d+):', fitness_plan)
    
    for day_match in day_matches:
        day_num = day_match.group(1)
        day_start_pos = day_match.start()
        
        # Find the next day or end of string
        next_day_match = re.search(r'DAY_\d+:', fitness_plan[day_start_pos + 1:])
        day_end_pos = day_start_pos + 1 + next_day_match.start() if next_day_match else len(fitness_plan)
        
        day_content = fitness_plan[day_start_pos:day_end_pos]
        print(f"DEBUG - plan: Processing day {day_num}, content length: {len(day_content)}")
        
        # Extract body parts
        body_parts_match = re.search(r'BODY_PARTS:\s*\[(.*?)\]', day_content, re.DOTALL)
        body_parts = body_parts_match.group(1) if body_parts_match else "Rest Day"
        
        # Clean up body parts string
        body_parts = body_parts.replace('"', '').replace("'", "").strip()
        print(f"DEBUG - plan: Day {day_num} body parts: {body_parts}")
        
        # Extract exercises
        exercises = []
        # More robust regex pattern that can handle various whitespace and formatting
        exercise_matches = re.finditer(r'- NAME:\s*(.*?)(?:\n|\r\n)\s*SETS:\s*(.*?)(?:\n|\r\n)\s*REPS:\s*(.*?)(?:\n|\r\n|$)', day_content, re.DOTALL)
        exercise_list = list(exercise_matches)
        print(f"DEBUG - plan: Day {day_num} has {len(exercise_list)} exercises")
        
        # If no exercises found with the first pattern, try an alternative pattern
        if len(exercise_list) == 0:
            print(f"DEBUG - plan: No exercises found with primary pattern, trying alternative")
            # Dump the day content for debugging
            print(f"DEBUG - Day content: {day_content[:200]}...")
            
            # Try alternative pattern
            exercise_matches = re.finditer(r'NAME:\s*(.*?)(?:\n|\r\n)\s*SETS:\s*(.*?)(?:\n|\r\n)\s*REPS:\s*(.*?)(?:\n|\r\n|$)', day_content, re.DOTALL)
            exercise_list = list(exercise_matches)
            print(f"DEBUG - plan: Day {day_num} has {len(exercise_list)} exercises with alternative pattern")
            
            # Reset the iterator
            exercise_matches = re.finditer(r'NAME:\s*(.*?)(?:\n|\r\n)\s*SETS:\s*(.*?)(?:\n|\r\n)\s*REPS:\s*(.*?)(?:\n|\r\n|$)', day_content, re.DOTALL)
        else:
            # Reset the iterator
            exercise_matches = re.finditer(r'- NAME:\s*(.*?)(?:\n|\r\n)\s*SETS:\s*(.*?)(?:\n|\r\n)\s*REPS:\s*(.*?)(?:\n|\r\n|$)', day_content, re.DOTALL)
        
        for ex_match in exercise_matches:
            exercises.append({
                'name': ex_match.group(1),
                'sets': ex_match.group(2),
                'reps': ex_match.group(3)
            })
        
        workout_plan[day_num] = {
            'body_parts': body_parts,
            'exercises': exercises
        }
    
    # Calculate BMI for display
    bmi = calculate_bmi(form_data.get('height_m', 0), form_data.get('weight_kg', 0))
    
    # Convert metric values back to imperial for display
    height_m = form_data.get('height_m', 0)
    total_inches = height_m / 0.0254
    height_feet = int(total_inches // 12)
    height_inches = int(total_inches % 12)
    
    weight_kg = form_data.get('weight_kg', 0)
    weight_lbs = round(weight_kg / 0.453592)
    
    goal_weight_kg = form_data.get('goal_weight_kg', 0)
    goal_weight_lbs = round(goal_weight_kg / 0.453592)
    
    return render_template('plan.html', 
                          workout_plan=workout_plan,
                          gender=form_data.get('gender', ''),
                          age=form_data.get('age', 30),
                          height_m=height_m,
                          height_feet=height_feet,
                          height_inches=height_inches,
                          weight_kg=weight_kg,
                          weight_lbs=weight_lbs,
                          goal_weight_kg=goal_weight_kg,
                          goal_weight_lbs=goal_weight_lbs,
                          goal=form_data.get('goal', ''),
                          activity_level=form_data.get('activity_level', 'moderately active'),
                          exercise_environment=form_data.get('exercise_environment', 'gym'),
                          bmi=bmi)

if __name__ == '__main__':
    app.run(debug=True)
