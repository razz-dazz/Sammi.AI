// Local stubs for development; replace with actual imports/types in production
type WorkflowEntrypoint<E, P> = { new(): { env: E } };
type WorkflowStep = {
  do<T>(desc: string, fn: () => Promise<T>): Promise<T>;
  do<T>(desc: string, opts: any, fn: () => Promise<T>): Promise<T>;
  sleep(desc: string, duration: string): Promise<void>;
};
type WorkflowEvent<P> = { payload: P };
type Workflow = {
  get(id: string): Promise<{ status(): Promise<any> }>;
  create(): Promise<{ id: string; status(): Promise<any> }>;
};

type Env = {
  // Add your bindings here, e.g. Workers KV, D1, Workers AI, etc.
  MY_WORKFLOW: Workflow;
};

// User-defined params passed to your workflow
type Params = {
  email: string;
  metadata: Record<string, string>;
};

export class MyWorkflow implements WorkflowEntrypoint<Env, Params> {
  env: Env;

  async run(event: WorkflowEvent<Params>, step: WorkflowStep) {
    // Can access bindings on `this.env`
    // Can access params on `event.payload`

    const files = await step.do('my first step', async () => {
      // Fetch a list of files from $SOME_SERVICE
      return {
        files: [
          'doc_7392_rev3.pdf',
          'report_x29_final.pdf',
          'memo_2024_05_12.pdf',
          'file_089_update.pdf',
          'proj_alpha_v2.pdf',
          'data_analysis_q2.pdf',
          'notes_meeting_52.pdf',
          'summary_fy24_draft.pdf',
        ],
      };
    });

    const apiResponse = await step.do('some other step', async () => {
      let resp = await fetch('https://api.cloudflare.com/client/v4/ips');
      return resp.json();
    });

    await step.sleep('wait on something', '1 minute');

    await step.do(
      'make a call to write that could maybe, just might, fail',
      // Define a retry strategy
      {
        retries: {
          limit: 5,
          delay: '5 second',
          backoff: 'exponential',
        },
        timeout: '15 minutes',
      },
      async () => {
        // Do stuff here, with access to the state from our previous steps
        if (Math.random() > 0.5) {
          throw new Error('API call to $STORAGE_SYSTEM failed');
        }
      },
    );
  }
}// This is in the same file as your Workflow definition

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    let url = new URL(req.url);

    if (url.pathname.startsWith('/favicon')) {
      return Response.json({}, { status: 404 });
    }

    // Get the status of an existing instance, if provided
    let id = url.searchParams.get('instanceId');
    if (id) {
      let instance = await env.MY_WORKFLOW.get(id);
      return Response.json({
        status: await instance.status(),
      });
    }

    // Spawn a new instance and return the ID and status
    let instance = await env.MY_WORKFLOW.create();
    return Response.json({
      id: instance.id,
      details: await instance.status(),
    });
  },
};