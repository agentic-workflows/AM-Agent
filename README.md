# AM-Agent

### Step by step to run it:

1. Start the Streaming Service (locally, it translates to having a Redis container running; in production, it translates to the S3M Streams Service)
2. Start the AC_Agent
3. [Optionally] Start the Flowcept Agent (to interact with the captured prov. data)
4. Start the Control Driver 
5. Run the HMI mock (which sends the initial message to trigger the execution)

