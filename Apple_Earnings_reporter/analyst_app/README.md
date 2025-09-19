# Apple Earnings Report Analyst

An AI-powered application that analyzes Apple's earnings reports and answers questions about financial performance. The analyst uses Google's Gemini AI to provide detailed, accurate responses based on the available quarterly reports.

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file and add your Gemini API key:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```
5. Place Apple earnings report PDFs in the `/analyst_app` folder
6. Run the application:
   ```bash
   python analyst_app.py
   ```

## How It Works

1. The application reads PDF earnings reports from the `/analyst_app` directory
2. At startup, it analyzes which quarters and years are covered in the reports
3. The AI analyst:
   - Formats large numbers in billions (B) or millions (M) for readability
   - Always specifies which quarter/year it's referring to
   - Only answers questions about available quarters
   - Provides clear responses when information is unavailable

## Example Prompts and Expected Results

### ✅ Good Prompts (With Expected Results)

1. **Basic Performance Question**
   ```
   Q: What was Apple's revenue in the latest quarter?
   A: Based on the Q3 2023 report, Apple's revenue was $81.8B for the quarter.
   ```

2. **Year-over-Year Comparison**
   ```
   Q: How did iPhone sales compare to the same quarter last year?
   A: In Q3 2023, iPhone revenue was $39.67B, showing a 2.4% decline from $40.67B in Q3 2022.
   ```

3. **Specific Metrics**
   ```
   Q: What was the gross margin percentage?
   A: For Q3 2023, Apple reported a gross margin of 44.5%, up from 43.3% in the same quarter last year.
   ```

4. **Geographic Performance**
   ```
   Q: How did Apple perform in different regions?
   A: In Q3 2023, Americas generated $35.38B, Europe $20.21B, Greater China $15.76B, Japan $4.82B, and Rest of Asia Pacific $5.63B in revenue.
   ```

### ❌ Out-of-Scope Prompts (With Expected Responses)

1. **Different Time Periods**
   ```
   Q: What was Apple's revenue in 2019?
   A: I apologize, but I only have information about Apple's earnings for [available quarters]. I cannot provide information about other periods.
   ```

2. **Other Companies**
   ```
   Q: How does this compare to Microsoft's performance?
   A: I apologize, but I only have information about Apple's earnings for [available quarters]. I cannot provide information about other companies.
   ```

3. **Future Predictions**
   ```
   Q: Will Apple's revenue grow next quarter?
   A: I apologize, but I can only analyze historical data from the available earnings reports for [available quarters]. I cannot make predictions about future performance.
   ```

4. **Non-Financial Questions**
   ```
   Q: What new products is Apple planning to launch?
   A: I can only provide information based on the financial data in the earnings reports for [available quarters]. Product roadmap information is not included in these reports.
   ```

## Best Practices for Asking Questions

1. **Be Specific**
   - Mention the time period if you're interested in specific quarters
   - Ask about specific metrics (revenue, margins, unit sales, etc.)

2. **Compare Within Available Data**
   - Ask for year-over-year comparisons within available quarters
   - Compare performance across different regions or product categories

3. **Focus on Financial Metrics**
   - Stick to financial and operational metrics
   - Ask about trends in the available data

## Limitations

- Only analyzes quarters included in the provided PDF reports
- Cannot make future predictions
- Cannot provide information about other companies
- Limited to financial and operational metrics in the earnings reports

## Troubleshooting

If you encounter issues:

1. Ensure PDF files are readable and contain earnings report data
2. Check that your API key is correctly set in the `.env` file
3. Make sure all dependencies are installed
4. Verify that PDFs are placed in the correct directory

## Contributing

Feel free to submit issues and enhancement requests!
