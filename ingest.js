import OpenAI from 'openai';
import pg from 'pg';
import { config } from 'dotenv'
import {
    formatVectorForPgvector,
    chunkText,
    chunkBySection,
} from "./utils.js"

config()

const client = new OpenAI({
    baseURL: process.env.AI_BASE_URL,
    apiKey: process.env.AI_API_KEY,
});

const EMBED_MODEL = process.env.EMBED_MODEL;

const pool = new pg.Pool({
    user: process.env.DB_USER,
    host: 'localhost',
    database: process.env.DB_NAME,
    password: process.env.DB_PASSWORD,
    port: 5432,
});

async function ingest() {
    const rawText = `
# Acme Technologies Human Resources Policy Manual

## Work From Home (WFH) Policy

Document ID: HR-WFH-2026-001

Effective Date: January 1, 2026

Last Updated: April 15, 2026

Policy Owner: Human Resources Department

---

## Purpose

The Work From Home (WFH) Policy provides eligible employees with the flexibility to perform their duties remotely while maintaining productivity, security, and collaboration.

This policy applies to all full-time and part-time employees of Acme Technologies.

---

## Eligibility

Employees are eligible to work from home if they meet all of the following requirements:

- Completed at least six (6) months of continuous employment.
- Successfully passed their probationary period.
- Have a performance rating of "Meets Expectations" or higher during the last evaluation.
- Their position is approved for remote work.
- Have a stable internet connection with a minimum speed of 50 Mbps download and 20 Mbps upload.

The following positions are generally eligible:

- Software Engineer
- QA Engineer
- UI/UX Designer
- Technical Writer
- DevOps Engineer
- Product Manager
- Business Analyst

The following positions are generally not eligible:

- Receptionist
- Facilities Staff
- Security Personnel
- Warehouse Associate

---

## Application Process

Employees requesting Work From Home must:

1. Submit a WFH Request Form.
2. Obtain approval from their immediate supervisor.
3. Obtain approval from the department manager.
4. Complete the Remote Security Awareness Training.
5. Sign the Remote Work Agreement.

Normal approval processing time is three (3) business days.

---

## Equipment

The company provides the following equipment:

- Company-issued laptop
- Monitor (up to 27 inches)
- Keyboard
- Mouse
- Headset
- Security token

Employees are responsible for:

- Internet service
- Electricity
- Workspace furniture

Additional equipment requests require approval from the IT Department.

---

## Internet Reimbursement

Eligible remote employees receive:

Monthly Internet Allowance:
USD 60

Employees must submit an internet bill every month before the 5th business day.

Late submissions will be processed during the next payroll cycle.

---

## Work Schedule

Regular work schedule:

Monday to Friday

9:00 AM – 6:00 PM

Lunch Break:
12:00 PM – 1:00 PM

Employees must be available during the company's Core Hours:

10:00 AM – 4:00 PM

Flexible working hours outside Core Hours may be approved by managers.

---

## Attendance

Employees working remotely must:

- Clock in using the TimeTrack Portal.
- Keep Microsoft Teams status updated.
- Attend all required meetings.
- Respond to work-related messages within one hour during Core Hours.

Repeated attendance violations may result in suspension of WFH privileges.

---

## Information Security

Employees must:

- Use company-issued devices only.
- Connect through the corporate VPN.
- Enable Multi-Factor Authentication (MFA).
- Lock their computer when away.
- Never share company passwords.
- Install security updates within seven (7) days of release.

The following are prohibited:

- Using public Wi-Fi without VPN.
- Storing confidential files on personal devices.
- Sharing company devices with family members.
- Printing confidential documents at home without approval.

---

## Data Classification

Company information is classified into:

Public

Internal

Confidential

Restricted

Restricted information may only be accessed using company-issued devices connected through VPN.

---

## Performance Expectations

Remote employees are expected to:

- Complete assigned tasks on time.
- Maintain at least 95% attendance.
- Meet quarterly performance goals.
- Participate in weekly team meetings.
- Respond to emails within four business hours.

---

## Health and Safety

Employees must maintain a workspace that is:

- Clean
- Well-lit
- Ergonomic
- Free from safety hazards

The company may request photographs of the workspace for ergonomic assessment.

---

## Temporary Remote Work

Employees may request temporary remote work for:

- Medical recovery
- Family emergencies
- Natural disasters
- Government travel restrictions

Temporary remote work is limited to 30 consecutive calendar days unless extended by HR.

---

## Exceptions

The Chief Human Resources Officer (CHRO) may approve exceptions to this policy.

Emergency exceptions may be approved verbally but must be documented within two business days.

---

## Frequently Asked Questions

Q: How long must an employee work before becoming eligible?

A: Six months.

Q: What is the minimum required internet speed?

A: 50 Mbps download and 20 Mbps upload.

Q: How much is the monthly internet allowance?

A: USD 60.

Q: Is public Wi-Fi allowed?

A: Yes, but only when connected through the corporate VPN.

Q: Who approves WFH requests?

A: The immediate supervisor and department manager.

Q: Can confidential documents be printed at home?

A: No, unless approved.

Q: What are the company's Core Hours?

A: 10:00 AM to 4:00 PM.

Q: How long can temporary remote work last?

A: Up to 30 consecutive calendar days unless extended by HR.

Q: Who can approve exceptions?

A: The Chief Human Resources Officer (CHRO).
    `;

    console.log('Chunking and Embedding text...');
    const chunks = chunkBySection(rawText);

    try {

        // Clear existing data for a fresh start
        await pool.query('DELETE FROM documents');
        console.log('Cleared existing documents.');

        let count = 0;
        for (const chunk of chunks) {
            if (chunk.trim().length < 10) continue;

            const response = await client.embeddings.create({
                model: EMBED_MODEL,
                input: chunk,
            });

            const embedding = response.data[0].embedding;
            const vectorString = formatVectorForPgvector(embedding);
            await pool.query(
                'INSERT INTO documents (text, embedding) VALUES ($1, $2)',
                [chunk, vectorString]
            );

            count++;
            console.log(`Stored chunk ${count}: "${chunk.substring(0, 50)}..."`);

            await new Promise(resolve => setTimeout(resolve, 100));
        }

        console.log(`\nIngestion complete! ${count} chunks stored in pgvector.`);
    } catch (err) {
        console.error(err);
        console.error('Error during ingestion:', err.message);
    } finally {
        await pool.end();
    }
}

ingest();