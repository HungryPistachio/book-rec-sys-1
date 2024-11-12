@app.post("/dice-explanation")
async def dice_explanation(request: Request):
    data = await request.json()
    recommendations = data.get("recommendations", [])
    logging.info("Received request for Dice explanation.")

    try:
        # Get the vectorized description
        vectorized_descriptions = recommendations[0]["vectorized_descriptions"]

        # Create a DataFrame with the correct number of columns
        input_data = pd.DataFrame([vectorized_descriptions], columns=fixed_vocabulary)

        # Ensure all feature values are numeric
        input_data = input_data.apply(pd.to_numeric)

        # Generate counterfactual explanation
        explanation = get_dice_explanation(dice, input_data)
        logging.info("Dice explanations generated successfully.")
        return JSONResponse(content=json.loads(explanation))
    except Exception as e:
        logging.error(f"Error in Dice explanation generation: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
