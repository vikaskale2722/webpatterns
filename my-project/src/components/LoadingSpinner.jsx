export default function LoadingSpinner({ message = "Processing..." }) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-gray-600">
        <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="mt-4 text-sm font-medium">{message}</p>
      </div>
    );
  }
  