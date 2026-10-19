# Qwiklabs Activity Tracking Step 1: Configure your environment and create the AlphaEvolve Cloud Storage bucket
# Runtime: Teacups (Cloud Run NsJail sandbox, read-only filesystem)

def check(handles:, maximum_score:, resources:)
  storage = handles['primary_project.StorageV1']
  project_id = resources.dig('primary_project', 'project_id') || resources.dig(:primary_project, :project_id) || ''
  ret_hash = { done: false, score: 0, message: 'Checking Cloud Storage for step 1', student_message: 'incomplete' }

  buckets = storage.list_buckets(project_id, freeze_args: true)&.items || []
  target_bucket = buckets.find { |b| b.name.to_s.include?('-alphaevolve-bucket') }
  if target_bucket
    ret_hash = {
      done: true,
      score: maximum_score,
      message: "Assessment Completed! Found bucket #{target_bucket.name}.",
      student_message: 'success'
    }
  else
    ret_hash[:message] = 'Expected Cloud Storage bucket ending in -alphaevolve-bucket not found.'
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
