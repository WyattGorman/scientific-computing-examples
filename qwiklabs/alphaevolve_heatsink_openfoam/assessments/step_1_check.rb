# Qwiklabs Activity Tracking Step 1: Configure your environment and create the AlphaEvolve Cloud Storage bucket
# Runtime: Teacups (Cloud Run NsJail sandbox, read-only filesystem)

def check(handles:, maximum_score:, resources:)
  storage = handles['project_0.StorageV1']
  project_id = resources.dig('project_0', 'project_id') || resources.dig(:project_0, :project_id) || ''
  expected_bucket = "#{project_id}-alphaevolve-bucket"
  ret_hash = {
    done: false,
    score: 0,
    message: "Checking for Cloud Storage bucket #{expected_bucket}.",
    student_message: 'incomplete'
  }

  buckets = storage.list_buckets(project_id, freeze_args: true)&.items || []
  target_bucket = buckets.find { |b| b.name.to_s == expected_bucket || b.name.to_s.include?('alphaevolve-bucket') }
  if target_bucket
    ret_hash = {
      done: true,
      score: maximum_score,
      message: "Assessment Completed! Found AlphaEvolve bucket #{target_bucket.name}.",
      student_message: 'success'
    }
  else
    ret_hash[:message] = "Expected Cloud Storage bucket #{expected_bucket} not found."
  end
  ret_hash
rescue StandardError => e
  {
    done: false,
    score: 0,
    message: "Activity tracking error in step 1: #{e.message.to_s.force_encoding('UTF-8').scrub}",
    student_message: 'incomplete'
  }
end

def step_1_check(handles:, maximum_score:, resources:)
  check(handles: handles, maximum_score: maximum_score, resources: resources)
end
