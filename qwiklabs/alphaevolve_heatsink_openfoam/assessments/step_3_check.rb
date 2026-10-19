# Qwiklabs Activity Tracking Step 3: Deploy the 3D heat sink optimization experiment blueprint and container images
# Runtime: Teacups (Cloud Run NsJail sandbox, read-only filesystem)

def check(handles:, maximum_score:, resources:)
  storage = handles['project_0.StorageV1']
  project_id = resources.dig('project_0', 'project_id') || resources.dig(:project_0, :project_id) || ''
  bucket_name = "#{project_id}-alphaevolve-bucket"
  ret_hash = {
    done: false,
    score: 0,
    message: 'Checking for heatsink-foam experiment deployment config in Cloud Storage.',
    student_message: 'incomplete'
  }

  objects = storage.list_objects(bucket_name, prefix: 'heatsink-foam/', freeze_args: true)&.items || []
  has_exp_env = objects.any? { |o| o.name.to_s == 'heatsink-foam/variables.env' }

  if has_exp_env
    ret_hash = {
      done: true,
      score: maximum_score,
      message: 'Assessment Completed! Found heatsink-foam/variables.env from alpha-evolve-experiment.yaml deployment.',
      student_message: 'success'
    }
  else
    ret_hash[:message] = 'Expected heatsink-foam/variables.env in the AlphaEvolve bucket after deploying alpha-evolve-experiment.yaml.'
  end
  ret_hash
rescue StandardError => e
  {
    done: false,
    score: 0,
    message: "Activity tracking error in step 3: #{e.message.to_s.force_encoding('UTF-8').scrub}",
    student_message: 'incomplete'
  }
end

def step_3_check(handles:, maximum_score:, resources:)
  check(handles: handles, maximum_score: maximum_score, resources: resources)
end
