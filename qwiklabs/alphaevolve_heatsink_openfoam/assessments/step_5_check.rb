# Qwiklabs Activity Tracking Step 5: Inspect the champion 3D cold-plate design and upload CHT plots to Cloud Storage
# Runtime: Teacups (Cloud Run NsJail sandbox, read-only filesystem)

def check(handles:, maximum_score:, resources:)
  storage = handles['project_0.StorageV1']
  project_id = resources.dig('project_0', 'project_id') || resources.dig(:project_0, :project_id) || ''
  bucket_name = "#{project_id}-alphaevolve-bucket"
  ret_hash = {
    done: false,
    score: 0,
    message: 'Checking for champion cold-plate summary JSON and OpenFOAM-14 CHT plots in Cloud Storage.',
    student_message: 'incomplete'
  }

  objects = storage.list_objects(bucket_name, prefix: 'heatsink-foam/results/', freeze_args: true)&.items || []
  object_names = objects.map { |o| o.name.to_s }
  required_artifacts = [
    'heatsink-foam/results/best_heatsink_summary.json',
    'heatsink-foam/results/optimized_heatsink.png'
  ]
  missing = required_artifacts.reject { |item| object_names.include?(item) }

  if missing.empty?
    ret_hash = {
      done: true,
      score: maximum_score,
      message: 'Assessment Completed! Verified best_heatsink_summary.json and optimized_heatsink.png in Cloud Storage.',
      student_message: 'success'
    }
  else
    ret_hash[:message] = "Missing expected visualization artifacts in gs://#{bucket_name}/: #{missing.join(', ')}"
  end
  ret_hash
rescue StandardError => e
  {
    done: false,
    score: 0,
    message: "Activity tracking error in step 5: #{e.message.to_s.force_encoding('UTF-8').scrub}",
    student_message: 'incomplete'
  }
end

def step_5_check(handles:, maximum_score:, resources:)
  check(handles: handles, maximum_score: maximum_score, resources: resources)
end
