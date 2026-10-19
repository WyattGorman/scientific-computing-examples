# Qwiklabs Activity Tracking Step 4: Submit the AlphaEvolve Controller job on Cloud Batch and monitor candidate results
# Runtime: Teacups (Cloud Run NsJail sandbox, read-only filesystem)

def check(handles:, maximum_score:, resources:)
  storage = handles['project_0.StorageV1']
  project_id = resources.dig('project_0', 'project_id') || resources.dig(:project_0, :project_id) || ''
  bucket_name = "#{project_id}-alphaevolve-bucket"
  ret_hash = {
    done: false,
    score: 0,
    message: 'Checking for AlphaEvolve 3D heat sink candidate evaluations in Cloud Storage.',
    student_message: 'incomplete'
  }

  objects = storage.list_objects(bucket_name, prefix: 'heatsink-foam/', freeze_args: true)&.items || []
  object_names = objects.map { |o| o.name.to_s }
  has_results = object_names.include?('heatsink-foam/current_experiment.json') ||
                object_names.include?('heatsink-foam/results.csv') ||
                object_names.any? { |n| n.start_with?('heatsink-foam/program_candidates/') } ||
                object_names.any? { |n| n.start_with?('heatsink-foam/archive/program_candidates/') }

  if has_results
    ret_hash = {
      done: true,
      score: maximum_score,
      message: 'Assessment Completed! Found AlphaEvolve 3D heat sink candidate evaluation artifacts in Cloud Storage.',
      student_message: 'success'
    }
  else
    ret_hash[:message] = 'Expected heatsink-foam/results.csv or candidate evaluation files in the AlphaEvolve bucket.'
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
