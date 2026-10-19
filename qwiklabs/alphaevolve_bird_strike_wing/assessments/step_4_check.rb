# Qwiklabs Activity Tracking Step 4: Submit the AlphaEvolve controller job on Cloud Batch and evaluate candidate wing designs
# Runtime: Teacups (Cloud Run NsJail sandbox, read-only filesystem)

def check(handles:, maximum_score:, resources:)
  storage = handles['primary_project.StorageV1']
  project_id = resources.dig('primary_project', 'project_id') || resources.dig(:primary_project, :project_id) || ''
  ret_hash = { done: false, score: 0, message: 'Checking Cloud Storage for step 4', student_message: 'incomplete' }

  buckets = storage.list_buckets(project_id, freeze_args: true)&.items || []
  target_bucket = buckets.find { |b| b.name.to_s.include?('-alphaevolve-bucket') }
  if target_bucket
    objs = storage.list_objects(target_bucket.name, prefix: 'bird-strike/results.csv', freeze_args: true)&.items || []
    if !objs.empty?
      ret_hash = {
        done: true,
        score: maximum_score,
        message: "Assessment Completed! Found bird-strike/results.csv in #{target_bucket.name}.",
        student_message: 'success'
      }
    else
      ret_hash[:message] = 'Expected bird-strike/results.csv in the AlphaEvolve Cloud Storage bucket.'
    end
  else
    ret_hash[:message] = 'Expected AlphaEvolve Cloud Storage bucket not found.'
  end
  ret_hash
rescue StandardError => e
  {
    done: false,
    score: 0,
    message: "Activity tracking error in step 4: #{e.message.to_s.force_encoding('UTF-8').scrub}",
    student_message: 'incomplete'
  }
end

def step_4_check(handles:, maximum_score:, resources:)
  check(handles: handles, maximum_score: maximum_score, resources: resources)
end
