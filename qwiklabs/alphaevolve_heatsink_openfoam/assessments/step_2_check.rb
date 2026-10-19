# Qwiklabs Activity Tracking Step 2: Deploy the AlphaEvolve base infrastructure with Cluster Toolkit
# Runtime: Teacups (Cloud Run NsJail sandbox, read-only filesystem)

def check(handles:, maximum_score:, resources:)
  storage = handles['project_0.StorageV1']
  project_id = resources.dig('project_0', 'project_id') || resources.dig(:project_0, :project_id) || ''
  bucket_name = "#{project_id}-alphaevolve-bucket"
  ret_hash = {
    done: false,
    score: 0,
    message: 'Checking for staged AlphaEvolve base infrastructure files in Cloud Storage.',
    student_message: 'incomplete'
  }

  objects = storage.list_objects(bucket_name, freeze_args: true)&.items || []
  object_names = objects.map { |o| o.name.to_s }
  has_infra_env = object_names.include?('config/variables-infra.env')
  has_notebook = object_names.include?('notebook/run_notebook.ipynb')

  if has_infra_env && has_notebook
    ret_hash = {
      done: true,
      score: maximum_score,
      message: 'Assessment Completed! Verified config/variables-infra.env and notebook/run_notebook.ipynb.',
      student_message: 'success'
    }
  else
    ret_hash[:message] = 'Expected config/variables-infra.env and notebook/run_notebook.ipynb in the AlphaEvolve bucket.'
  end
  ret_hash
rescue StandardError => e
  {
    done: false,
    score: 0,
    message: "Activity tracking error in step 2: #{e.message.to_s.force_encoding('UTF-8').scrub}",
    student_message: 'incomplete'
  }
end

def step_2_check(handles:, maximum_score:, resources:)
  check(handles: handles, maximum_score: maximum_score, resources: resources)
end
